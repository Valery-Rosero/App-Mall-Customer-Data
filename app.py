"""Aplicación Streamlit: segmentación de clientes del centro comercial con K-Means."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

BASE = Path(__file__).parent
SEMILLA = 42
SIMBOLOS = {"o": "circle", "s": "square", "D": "diamond", "^": "triangle-up", "v": "triangle-down"}

DESCRIPCIONES = {
    "alto": "Ingreso alto y gasto alto. Son los clientes más rentables: compran seguido y gastan mucho.",
    "potencial": "Ingreso alto pero gasto bajo. Tienen capacidad de compra que hoy no se está aprovechando.",
    "promedio": "Ingreso y gasto medios. Es la base estable y más numerosa del negocio.",
    "impulsivo": "Ingreso bajo pero gasto alto. Suelen ser jóvenes, activos y sensibles a tendencias.",
    "bajo": "Ingreso bajo y gasto bajo. Compran poco y principalmente lo esencial.",
}
ESTRATEGIAS = {
    "alto": ["Programa VIP con beneficios exclusivos", "Atención personalizada (personal shopper)",
             "Productos premium y lanzamientos anticipados", "Monitoreo de señales de abandono"],
    "potencial": ["Investigar por qué no compran (encuestas)", "Propuestas de calidad y exclusividad, no descuentos",
                  "Servicios premium y experiencias", "Incentivos de bienvenida atados a la primera compra alta"],
    "promedio": ["Venta cruzada y combos", "Programa de puntos por frecuencia",
                 "Promociones de temporada", "Notificaciones personalizadas en la app"],
    "impulsivo": ["Redes sociales e influencers", "Flash sales y novedades",
                  "Pago en cuotas responsable", "Fidelizar para cuando sus ingresos crezcan"],
    "bajo": ["Comunicación masiva de bajo costo", "Outlets y días de descuento",
             "Productos esenciales y marcas propias", "Baja inversión en marketing personalizado"],
}

st.set_page_config(page_title="Segmentación de clientes · K-Means", page_icon="🛍️", layout="wide")


# ------------------------------------------------------------------ datos y modelo
@st.cache_data
def cargar_datos():
    df = pd.read_csv(BASE / "Mall_Customers.csv")
    return df.rename(columns={"CustomerID": "ID", "Gender": "Genero", "Age": "Edad",
                              "Annual Income (k$)": "Ingreso", "Spending Score (1-100)": "Gasto"})


@st.cache_resource
def cargar_modelo():
    return joblib.load(BASE / "modelo" / "kmeans_clientes.joblib")


@st.cache_data
def curvas_k(X_esc, k_max=10):
    inercias, siluetas, db = [], [], []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, n_init=50, random_state=SEMILLA).fit(X_esc)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(X_esc, km.labels_) if k > 1 else np.nan)
        db.append(davies_bouldin_score(X_esc, km.labels_) if k > 1 else np.nan)
    return pd.DataFrame({"K": range(1, k_max + 1), "Inercia": inercias, "Silueta": siluetas, "Davies-Bouldin": db})


df = cargar_datos()
modelo = cargar_modelo()
scaler, kmeans, FEATURES = modelo["scaler"], modelo["kmeans"], modelo["features"]
MAPA, PERFILES = modelo["mapa_perfil"], modelo["perfiles"]
K = kmeans.n_clusters

X_esc = scaler.transform(df[FEATURES])
df["Cluster"] = kmeans.predict(X_esc)
df["Clave"] = df["Cluster"].map(MAPA)
df["Perfil"] = df["Clave"].map(lambda c: PERFILES[c]["nombre"])
centroides = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES)
centroides["Clave"] = [MAPA[c] for c in range(K)]


def clasificar(ingreso, gasto):
    x = scaler.transform(pd.DataFrame([[ingreso, gasto]], columns=FEATURES))
    cluster = int(kmeans.predict(x)[0])
    distancias = np.linalg.norm(kmeans.cluster_centers_ - x, axis=1)
    return cluster, MAPA[cluster], distancias


def figura_clusters(nuevo=None):
    fig = go.Figure()
    for clave, info in PERFILES.items():
        sub = df[df["Clave"] == clave]
        fig.add_trace(go.Scatter(
            x=sub["Ingreso"], y=sub["Gasto"], mode="markers", name=f"{info['nombre']} ({len(sub)})",
            marker={"color": info["color"], "symbol": SIMBOLOS[info["marcador"]], "size": 10,
                    "line": {"width": 1.2, "color": "white"}},
            customdata=sub[["ID", "Edad", "Genero"]],
            hovertemplate="<b>Cliente %{customdata[0]}</b><br>Ingreso: %{x} k$<br>Gasto: %{y}"
                          "<br>Edad: %{customdata[1]} · %{customdata[2]}<extra>" + info["nombre"] + "</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=centroides["Ingreso"], y=centroides["Gasto"], mode="markers+text", name="Centroides",
        marker={"symbol": "x", "size": 16, "color": "#0b0b0b", "line": {"width": 2, "color": "white"}},
        text=[PERFILES[c]["nombre"].replace("Clientes ", "").capitalize() for c in centroides["Clave"]],
        textposition="top right", textfont={"size": 13},
        hovertemplate="<b>Centroide %{text}</b><br>Ingreso: %{x:.1f} k$<br>Gasto: %{y:.1f}<extra></extra>",
    ))
    if nuevo is not None:
        fig.add_trace(go.Scatter(
            x=[nuevo[0]], y=[nuevo[1]], mode="markers", name="Nuevo cliente",
            marker={"symbol": "star", "size": 26, "color": "#eb6834", "line": {"width": 2, "color": "white"}},
            hovertemplate="<b>Nuevo cliente</b><br>Ingreso: %{x} k$<br>Gasto: %{y}<extra></extra>",
        ))
    fig.update_layout(height=560, xaxis_title="Ingreso anual (k$)", yaxis_title="Puntaje de gasto (1-100)",
                      legend={"title": "Segmento"}, margin={"t": 30})
    return fig


def figura_curva(x, y, marcado, titulo_y, color="#2a78d6"):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", line={"color": color, "width": 2.5},
                               marker={"size": 9, "line": {"width": 2, "color": "white"}},
                               hovertemplate="K = %{x}<br>" + titulo_y + ": %{y:.3f}<extra></extra>"))
    y_marcado = y[list(x).index(marcado)]
    fig.add_trace(go.Scatter(x=[marcado], y=[y_marcado], mode="markers", showlegend=False, hoverinfo="skip",
                             marker={"size": 22, "color": "rgba(0,0,0,0)", "line": {"width": 3, "color": "#eb6834"}}))
    fig.add_annotation(x=marcado, y=y_marcado, text=f"K = {marcado}", showarrow=True, arrowhead=2, ax=50, ay=-40)
    fig.update_layout(height=420, showlegend=False, xaxis={"title": "Número de clusters (K)", "dtick": 1},
                      yaxis_title=titulo_y, margin={"t": 30})
    return fig


# ------------------------------------------------------------------ encabezado
st.title("🛍️ Segmentación de clientes con K-Means")
st.caption("Mall Customers Dataset · variables: ingreso anual y puntaje de gasto · modelo estandarizado con StandardScaler")

m = modelo["metricas"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes analizados", len(df))
c2.metric("Número de clusters (K)", K)
c3.metric("Coeficiente de silueta", f"{m['silueta']:.3f}")
c4.metric("Davies-Bouldin", f"{m['davies_bouldin']:.3f}", help="Menor es mejor")

tab_nuevo, tab_clusters, tab_codo, tab_silueta, tab_datos, tab_lote = st.tabs(
    ["🧑 Clasificar cliente", "🎯 Clusters", "📉 Método del codo", "📈 Silueta", "🔎 Datos", "📁 Clasificación masiva"])

# ------------------------------------------------------------------ clasificar nuevo cliente
with tab_nuevo:
    st.subheader("Ingresar un nuevo cliente")
    col_form, col_res = st.columns([1, 2], gap="large")
    with col_form:
        with st.form("nuevo_cliente"):
            ingreso = st.number_input("Ingreso anual (k$)", min_value=1.0, max_value=300.0, value=70.0, step=1.0)
            gasto = st.slider("Puntaje de gasto (1-100)", 1, 100, 60)
            st.caption("Datos de referencia (no se usan en el modelo):")
            edad = st.number_input("Edad", min_value=18, max_value=100, value=30)
            genero = st.radio("Género", ["Femenino", "Masculino"], horizontal=True)
            enviado = st.form_submit_button("Clasificar cliente", type="primary", width="stretch")

    cluster, clave, distancias = clasificar(ingreso, gasto)
    info = PERFILES[clave]
    with col_res:
        if not enviado:
            st.info("Ajusta los datos y presiona **Clasificar cliente**. Abajo se muestra la vista previa.")
        st.markdown(
            f"<div style='border-left:6px solid {info['color']};padding:0.6rem 1rem;border-radius:6px;"
            f"background:rgba(127,127,127,0.08)'><div style='font-size:0.9rem;opacity:0.75'>Segmento asignado "
            f"(cluster {cluster})</div><div style='font-size:1.7rem;font-weight:700'>{info['nombre']}</div>"
            f"<div>{DESCRIPCIONES[clave]}</div></div>", unsafe_allow_html=True)
        st.markdown("**Estrategias de marketing recomendadas**")
        st.markdown("\n".join(f"- {e}" for e in ESTRATEGIAS[clave]))

        if not (df["Ingreso"].min() <= ingreso <= df["Ingreso"].max()):
            st.warning(f"El ingreso ingresado está fuera del rango de entrenamiento "
                       f"({df['Ingreso'].min()}–{df['Ingreso'].max()} k$). La asignación es una extrapolación.")

        dist = pd.DataFrame({"Segmento": [PERFILES[MAPA[c]]["nombre"] for c in range(K)],
                             "Distancia al centroide (z)": distancias}).sort_values("Distancia al centroide (z)")
        st.dataframe(dist, hide_index=True, width="stretch",
                     column_config={"Distancia al centroide (z)": st.column_config.ProgressColumn(
                         format="%.2f", min_value=0, max_value=float(dist.iloc[:, 1].max()))})

    st.plotly_chart(figura_clusters((ingreso, gasto)), width="stretch")

# ------------------------------------------------------------------ clusters
with tab_clusters:
    st.subheader("Clusters generados")
    st.plotly_chart(figura_clusters(), width="stretch")

    resumen = df.groupby("Clave").agg(
        Clientes=("ID", "size"), Edad=("Edad", "mean"), Ingreso=("Ingreso", "mean"), Gasto=("Gasto", "mean"),
        Mujeres=("Genero", lambda s: (s == "Female").mean() * 100)).reindex(list(PERFILES))
    resumen.insert(0, "Segmento", [PERFILES[c]["nombre"] for c in resumen.index])
    resumen.insert(2, "% del total", resumen["Clientes"] / len(df) * 100)
    st.dataframe(resumen, hide_index=True, width="stretch", column_config={
        "% del total": st.column_config.NumberColumn(format="%.1f %%"),
        "Edad": st.column_config.NumberColumn("Edad media", format="%.1f"),
        "Ingreso": st.column_config.NumberColumn("Ingreso medio (k$)", format="%.1f"),
        "Gasto": st.column_config.NumberColumn("Gasto medio", format="%.1f"),
        "Mujeres": st.column_config.NumberColumn("% mujeres", format="%.0f %%")})

    st.markdown("#### Perfiles y estrategias")
    cols = st.columns(len(PERFILES))
    for col, (clave_p, info_p) in zip(cols, PERFILES.items()):
        with col:
            st.markdown(f"<div style='height:6px;background:{info_p['color']};border-radius:3px'></div>",
                        unsafe_allow_html=True)
            st.markdown(f"**{info_p['nombre']}**")
            st.caption(DESCRIPCIONES[clave_p])
            st.markdown("\n".join(f"- {e}" for e in ESTRATEGIAS[clave_p]))
    st.success("**Mayor oportunidad de crecimiento: clientes con potencial.** Tienen el ingreso más alto "
               "(≈ 88 k$) y el gasto más bajo (≈ 17). Si subieran su gasto al nivel promedio, casi lo triplicarían.")

# ------------------------------------------------------------------ método del codo
curvas = curvas_k(X_esc)
with tab_codo:
    st.subheader("Método del codo")
    st.plotly_chart(figura_curva(curvas["K"], curvas["Inercia"].tolist(), K, "Inercia (WCSS)"),
                    width="stretch")
    tabla = curvas[["K", "Inercia"]].copy()
    tabla["Reducción vs K-1 (%)"] = -tabla["Inercia"].pct_change() * 100
    c_a, c_b = st.columns([1, 1.4])
    c_a.dataframe(tabla, hide_index=True, width="stretch",
                  column_config={"Inercia": st.column_config.NumberColumn(format="%.2f"),
                                 "Reducción vs K-1 (%)": st.column_config.NumberColumn(format="%.1f")})
    c_b.markdown(
        "La **inercia** es la suma de distancias al cuadrado de cada cliente a su centroide: entre más cerca de 0, "
        "más compactos son los grupos. Siempre baja al aumentar K, así que se busca el **codo**, el punto desde el "
        "cual agregar clusters ya casi no la reduce.\n\n"
        "De K = 4 a K = 5 la inercia todavía cae un **39.8 %**. Desde K = 6 la reducción es pequeña "
        "y la curva se aplana, así que el codo está en **K = 5**.")

# ------------------------------------------------------------------ silueta
with tab_silueta:
    st.subheader("Coeficiente de silueta")
    sil = curvas.dropna()
    k_sil = int(sil.loc[sil["Silueta"].idxmax(), "K"])
    st.plotly_chart(figura_curva(sil["K"], sil["Silueta"].tolist(), k_sil, "Silueta"), width="stretch")
    c_a, c_b = st.columns([1, 1.4])
    c_a.dataframe(sil[["K", "Silueta", "Davies-Bouldin"]], hide_index=True, width="stretch",
                  column_config={"Silueta": st.column_config.NumberColumn(format="%.4f"),
                                 "Davies-Bouldin": st.column_config.NumberColumn(format="%.3f")})
    c_b.markdown(
        f"La silueta compara la cohesión de cada cliente con su grupo y su separación del grupo más cercano. "
        f"Va de −1 a 1: cerca de **1** significa grupos compactos y bien separados, cerca de **0** significa grupos "
        f"que se superponen y un valor **negativo** significa clientes mal asignados.\n\n"
        f"El máximo está en **K = {k_sil}** (s = {sil['Silueta'].max():.3f}), igual que en el método del codo. "
        f"El índice Davies-Bouldin también es mínimo en ese K.")

# ------------------------------------------------------------------ datos
with tab_datos:
    st.subheader("Exploración de los datos")
    c_a, c_b = st.columns(2)
    c_a.markdown("**Estadísticas descriptivas**")
    c_a.dataframe(df[["Edad", "Ingreso", "Gasto"]].describe().T, width="stretch")
    c_b.markdown("**Correlaciones de Pearson**")
    c_b.dataframe(df[["Edad", "Ingreso", "Gasto"]].corr().style.format("{:.2f}"), width="stretch")
    st.markdown("**Clientes con su segmento asignado**")
    st.dataframe(df[["ID", "Genero", "Edad", "Ingreso", "Gasto", "Perfil"]], hide_index=True,
                 width="stretch", height=360)

# ------------------------------------------------------------------ clasificación masiva
with tab_lote:
    st.subheader("Clasificar varios clientes desde un CSV")
    st.markdown("Sube un CSV con las columnas `Ingreso` y `Gasto` "
                "(o `Annual Income (k$)` y `Spending Score (1-100)` como en el dataset original).")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        lote = pd.read_csv(archivo).rename(columns={"Annual Income (k$)": "Ingreso",
                                                    "Spending Score (1-100)": "Gasto"})
        faltantes = [c for c in FEATURES if c not in lote.columns]
        if faltantes:
            st.error(f"Faltan las columnas: {', '.join(faltantes)}")
        else:
            validos = lote[FEATURES].notna().all(axis=1)
            etiquetas = kmeans.predict(scaler.transform(lote.loc[validos, FEATURES]))
            lote.loc[validos, "Segmento"] = [PERFILES[MAPA[int(c)]]["nombre"] for c in etiquetas]
            if (~validos).any():
                st.warning(f"{(~validos).sum()} filas tienen valores vacíos y no se clasificaron.")
            st.dataframe(lote, hide_index=True, width="stretch")
            st.download_button("Descargar resultados", lote.to_csv(index=False).encode("utf-8"),
                               "clientes_segmentados.csv", "text/csv")
