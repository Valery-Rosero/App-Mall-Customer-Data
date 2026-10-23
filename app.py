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
REPOSITORIO = "https://github.com/Valery-Rosero/App-Mall-Customer-Data"
INFORME_PDF = BASE / "Informe_APA_Segmentacion_KMeans.pdf"
INFORME_DOCX = BASE / "Informe_APA_Segmentacion_KMeans.docx"
SIMBOLOS = {"o": "circle", "s": "square", "D": "diamond", "^": "triangle-up", "v": "triangle-down"}

# Paleta Calm: Sage, Moss, Mist, Ivory y Sand, más tintas neutras
SAGE, MOSS, MIST, IVORY, SAND = "#6B7F6E", "#A7B8A7", "#D9E1D9", "#F4EFE8", "#C9BDAF"
TINTA, TINTA_2, TENUE, SUPERFICIE, REJILLA = "#1F2A22", "#5C6660", "#8A918B", "#FBF9F5", "#E6E0D6"
ACENTO = "#815200"

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

# Íconos de línea (trazos estilo Lucide) para los bloques HTML
ICONOS = {
    "bolsa": '<path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4z"/><path d="M3 6h18"/>'
             '<path d="M16 10a4 4 0 0 1-8 0"/>',
    "usuarios": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
                '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "capas": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "pulso": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "barras": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "github": '<path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 '
              '0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 '
              '5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/>',
    "flecha": '<path d="M7 17 17 7"/><path d="M7 7h10v10"/>',
}


def icono(nombre, tam=20, color="currentColor", grosor=1.75):
    return (f'<svg width="{tam}" height="{tam}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{grosor}" stroke-linecap="round" stroke-linejoin="round">{ICONOS[nombre]}</svg>')


st.set_page_config(page_title="Segmentación de clientes · K-Means", page_icon=":material/donut_small:",
                   layout="wide")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.5rem; max-width: 1200px; }}
  h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}
  .encabezado {{ text-align: center; margin-bottom: 1.8rem; }}
  .encabezado .marca {{ display:inline-flex; width:52px; height:52px; border-radius:50%; background:{MIST};
                        align-items:center; justify-content:center; color:{SAGE}; margin-bottom:0.8rem; }}
  .encabezado h1 {{ font-weight:300; font-size:2.6rem; letter-spacing:0.02em; margin:0; padding:0; color:{TINTA}; }}
  .encabezado .sub {{ text-transform:uppercase; letter-spacing:0.32em; font-size:0.72rem; color:{TINTA_2};
                      margin-top:0.6rem; }}
  .repo {{ display:inline-flex; align-items:center; gap:0.55rem; margin-top:1.1rem; padding:0.55rem 1.1rem;
           border-radius:999px; background:{SUPERFICIE}; color:{TINTA} !important; text-decoration:none !important;
           font-size:0.9rem; font-weight:500; border:1px solid #DDD5C9; transition:background .2s ease; }}
  .repo:hover {{ background:{MIST}; }}
  .pie {{ text-align:center; color:{TINTA_2}; font-size:0.8rem; margin:2.5rem 0 1rem; line-height:1.8; }}
  .pie a {{ color:{SAGE} !important; font-weight:500; }}
  .etiqueta {{ text-transform:uppercase; letter-spacing:0.2em; font-size:0.7rem; font-weight:500;
               color:{TINTA_2}; margin:0.4rem 0 0.9rem; }}
  .etiqueta b {{ color:{SAGE}; font-weight:600; margin-right:0.5rem; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:1rem; margin-bottom:1.6rem; }}
  .kpi {{ background:{SUPERFICIE}; border-radius:20px; padding:1.1rem 1.2rem; display:flex; gap:0.9rem;
          align-items:center; box-shadow:0 1px 2px rgba(31,42,34,0.04), 0 8px 24px rgba(31,42,34,0.05); }}
  .kpi .ico {{ flex:none; width:44px; height:44px; border-radius:50%; background:{MIST}; color:{SAGE};
               display:flex; align-items:center; justify-content:center; }}
  .kpi .lbl {{ text-transform:uppercase; letter-spacing:0.14em; font-size:0.66rem; color:{TINTA_2}; }}
  .kpi .val {{ font-size:1.55rem; font-weight:500; color:{TINTA}; line-height:1.2; }}
  [class*="st-key-tarjeta"] {{ background:{SUPERFICIE}; border-radius:24px !important; border:none !important;
          box-shadow:0 1px 2px rgba(31,42,34,0.04), 0 10px 30px rgba(31,42,34,0.06); padding:1.4rem 1.5rem; }}
  .stTabs [role="tablist"] {{ background:{SUPERFICIE}; border-radius:999px; padding:6px; gap:4px; border:none;
          box-shadow:0 1px 2px rgba(31,42,34,0.04), 0 8px 24px rgba(31,42,34,0.05); flex-wrap:nowrap;
          overflow-x:auto; scrollbar-width:none; width:fit-content; max-width:100%; }}
  .stTabs [role="tablist"]::-webkit-scrollbar {{ display:none; }}
  .stTabs [role="tablist"]::before, .stTabs [role="tablist"]::after,
  .stTabs [role="tab"]::before, .stTabs [role="tab"]::after,
  .stTabs .react-aria-SelectionIndicator {{ display:none !important; }}
  .stTabs [role="tab"] {{ white-space:nowrap; flex:none; }}
  .stTabs > div:first-child {{ border:none !important; box-shadow:none !important; }}
  .stTabs [role="tab"] {{ border-radius:999px; padding:0.5rem 1.15rem; height:auto; margin:0;
          border:none !important; color:{TINTA_2}; transition:background .2s ease; }}
  .stTabs [role="tab"]:hover {{ background:{MIST}; }}
  .stTabs [role="tab"][aria-selected="true"] {{ background:{SAGE}; }}
  .stTabs [role="tab"][aria-selected="true"] p, .stTabs [role="tab"][aria-selected="true"] span {{ color:#FFFFFF; }}
  .stTabs [role="tabpanel"] {{ padding-top:1.4rem; }}
  .resultado {{ background:#E9EEE8; border-radius:20px; padding:1.1rem 1.3rem; display:flex; gap:1rem;
                align-items:flex-start; }}
  .resultado .circ {{ flex:none; width:46px; height:46px; border-radius:50%; display:flex; align-items:center;
                      justify-content:center; }}
  .resultado .chip {{ display:inline-block; text-transform:uppercase; letter-spacing:0.14em; font-size:0.64rem;
                      color:{TINTA_2}; background:{SUPERFICIE}; border-radius:999px; padding:0.15rem 0.6rem; }}
  .resultado .nom {{ font-size:1.45rem; font-weight:600; color:{TINTA}; margin:0.25rem 0 0.15rem; }}
  .resultado .desc {{ color:{TINTA_2}; }}
  .lista {{ list-style:none; padding:0 !important; margin:0.3rem 0 0 !important; }}
  .lista li {{ display:flex; gap:0.6rem; align-items:center; padding:0.32rem 0; margin:0 !important;
               color:{TINTA}; }}
  .lista li svg {{ flex:none; }}
  .perfil {{ background:{SUPERFICIE}; border-radius:20px; padding:1.1rem 1.1rem 0.8rem; height:100%;
             box-shadow:0 1px 2px rgba(31,42,34,0.04), 0 8px 24px rgba(31,42,34,0.05); }}
  .perfil .barra {{ height:6px; border-radius:999px; margin-bottom:0.8rem; }}
  .perfil .nom {{ font-weight:600; color:{TINTA}; }}
  .perfil .pct {{ font-size:0.78rem; color:{TINTA_2}; margin-bottom:0.4rem; }}
  .perfil .desc {{ font-size:0.86rem; color:{TINTA_2}; margin-bottom:0.4rem; }}
  .perfil .lista li {{ font-size:0.86rem; padding:0.2rem 0; align-items:flex-start; }}
</style>
""", unsafe_allow_html=True)


def etiqueta(numero, texto):
    st.markdown(f'<div class="etiqueta"><b>{numero:02d}.</b>{texto}</div>', unsafe_allow_html=True)


def lista_check(items, color=SAGE):
    filas = "".join(f"<li>{icono('check', 16, color, 2.2)}<span>{e}</span></li>" for e in items)
    return f'<ul class="lista">{filas}</ul>'


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


def estilo(fig, alto):
    """Aplica el estilo Calm a una figura de plotly."""
    eje = {"gridcolor": REJILLA, "zeroline": False, "linecolor": SAND, "tickfont": {"color": TENUE},
           "title": {"font": {"color": TINTA_2}}, "automargin": True}
    fig.update_layout(
        height=alto, font={"family": "Inter, sans-serif", "color": TINTA, "size": 13},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SUPERFICIE, margin={"t": 30, "l": 60, "r": 10, "b": 50},
        hoverlabel={"bgcolor": "#FFFFFF", "bordercolor": REJILLA, "font": {"family": "Inter", "color": TINTA}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0,
                "font": {"color": TINTA_2}},
    )
    fig.update_xaxes(**eje)
    fig.update_yaxes(**eje)
    return fig


def figura_clusters(nuevo=None):
    fig = go.Figure()
    for clave, info in PERFILES.items():
        sub = df[df["Clave"] == clave]
        fig.add_trace(go.Scatter(
            x=sub["Ingreso"], y=sub["Gasto"], mode="markers", name=f"{info['nombre']} ({len(sub)})",
            marker={"color": info["color"], "symbol": SIMBOLOS[info["marcador"]], "size": 10,
                    "line": {"width": 1.2, "color": SUPERFICIE}},
            customdata=sub[["ID", "Edad", "Genero"]],
            hovertemplate="<b>Cliente %{customdata[0]}</b><br>Ingreso: %{x} k$<br>Gasto: %{y}"
                          "<br>Edad: %{customdata[1]} · %{customdata[2]}<extra>" + info["nombre"] + "</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=centroides["Ingreso"], y=centroides["Gasto"], mode="markers+text", name="Centroides",
        marker={"symbol": "x", "size": 15, "color": TINTA, "line": {"width": 2, "color": SUPERFICIE}},
        text=[PERFILES[c]["nombre"].replace("Clientes ", "").capitalize() for c in centroides["Clave"]],
        textposition="top right", textfont={"size": 12, "color": TINTA},
        hovertemplate="<b>Centroide %{text}</b><br>Ingreso: %{x:.1f} k$<br>Gasto: %{y:.1f}<extra></extra>",
    ))
    if nuevo is not None:
        fig.add_trace(go.Scatter(
            x=[nuevo[0]], y=[nuevo[1]], mode="markers+text", name="Nuevo cliente", text=["Nuevo cliente"],
            textposition="bottom center", textfont={"size": 12, "color": TINTA},
            marker={"symbol": "star", "size": 26, "color": TINTA, "line": {"width": 2, "color": SUPERFICIE}},
            hovertemplate="<b>Nuevo cliente</b><br>Ingreso: %{x} k$<br>Gasto: %{y}<extra></extra>",
        ))
    fig.update_layout(xaxis_title="Ingreso anual (k$)", yaxis_title="Puntaje de gasto (1-100)")
    return estilo(fig, 560)


def figura_curva(x, y, marcado, titulo_y):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", line={"color": SAGE, "width": 2.5},
                               marker={"size": 9, "color": SAGE, "line": {"width": 2, "color": SUPERFICIE}},
                               hovertemplate="K = %{x}<br>" + titulo_y + ": %{y:.3f}<extra></extra>"))
    y_marcado = y[list(x).index(marcado)]
    fig.add_trace(go.Scatter(x=[marcado], y=[y_marcado], mode="markers", hoverinfo="skip",
                             marker={"size": 22, "color": "rgba(0,0,0,0)", "line": {"width": 2.5, "color": ACENTO}}))
    fig.add_annotation(x=marcado, y=y_marcado, text=f"K = {marcado}", showarrow=True, arrowhead=2,
                       arrowcolor=TINTA_2, ax=50, ay=-40, font={"color": TINTA})
    fig.update_layout(showlegend=False, xaxis={"title": "Número de clusters (K)", "dtick": 1}, yaxis_title=titulo_y)
    return estilo(fig, 420)


def grafica(fig):
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})


# ------------------------------------------------------------------ encabezado
st.markdown(f"""
<div class="encabezado">
  <div class="marca">{icono('bolsa', 24)}</div>
  <h1>Segmentación de clientes</h1>
  <div class="sub">K-Means · Mall Customers · Ingreso y gasto</div>
  <a class="repo" href="{REPOSITORIO}" target="_blank" rel="noopener">{icono('github', 18)}
     Ver repositorio en GitHub {icono('flecha', 14)}</a>
</div>
""", unsafe_allow_html=True)

m = modelo["metricas"]
kpis = [("usuarios", "Clientes analizados", len(df)), ("capas", "Número de clusters", K),
        ("pulso", "Coeficiente de silueta", f"{m['silueta']:.3f}"),
        ("barras", "Davies-Bouldin", f"{m['davies_bouldin']:.3f}")]
st.markdown('<div class="kpis">' + "".join(
    f'<div class="kpi"><div class="ico">{icono(i)}</div><div><div class="lbl">{l}</div>'
    f'<div class="val">{v}</div></div></div>' for i, l, v in kpis) + "</div>", unsafe_allow_html=True)

tab_nuevo, tab_clusters, tab_codo, tab_silueta, tab_datos, tab_lote, tab_informe = st.tabs([
    ":material/person_search: Clasificar cliente", ":material/scatter_plot: Clusters",
    ":material/show_chart: Método del codo", ":material/insights: Silueta",
    ":material/table_chart: Datos", ":material/upload_file: Clasificación masiva",
    ":material/description: Informe"])

# ------------------------------------------------------------------ clasificar nuevo cliente
with tab_nuevo:
    col_form, col_res = st.columns([1, 2], gap="large")
    with col_form:
        with st.container(key="tarjeta_formulario"):
            etiqueta(1, "Nuevo cliente")
            ingreso = st.number_input("Ingreso anual (k$)", min_value=1.0, max_value=300.0, value=70.0, step=1.0)
            gasto = st.slider("Puntaje de gasto (1-100)", 1, 100, 60)
            st.caption("Datos de referencia (no se usan en el modelo)")
            edad = st.number_input("Edad", min_value=18, max_value=100, value=30)
            genero = st.segmented_control("Género", ["Femenino", "Masculino"], default="Femenino") or "Femenino"
            st.caption(":material/bolt: La clasificación se actualiza automáticamente.")

    cluster, clave, distancias = clasificar(ingreso, gasto)
    info = PERFILES[clave]
    edad_segmento = df.loc[df["Clave"] == clave, "Edad"].mean()
    with col_res:
        with st.container(key="tarjeta_resultado"):
            etiqueta(2, "Segmento asignado")
            st.markdown(f"""
            <div class="resultado">
              <div class="circ" style="background:{info['color']}">{icono('check', 22, '#FFFFFF', 2.4)}</div>
              <div>
                <span class="chip">Cluster {cluster}</span>
                <div class="nom">{info['nombre']}</div>
                <div class="desc">{DESCRIPCIONES[clave]}</div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.caption(f"Cliente {genero.lower()} de {edad} años · edad media del segmento: "
                       f"{edad_segmento:.1f} años")

            if not (df["Ingreso"].min() <= ingreso <= df["Ingreso"].max()):
                st.warning(f"El ingreso está fuera del rango de entrenamiento ({df['Ingreso'].min()}–"
                           f"{df['Ingreso'].max()} k$). La asignación es una extrapolación.",
                           icon=":material/warning:")

            c_est, c_dist = st.columns([1, 1.15], gap="medium")
            with c_est:
                etiqueta(3, "Estrategias")
                st.markdown(lista_check(ESTRATEGIAS[clave]), unsafe_allow_html=True)
            with c_dist:
                etiqueta(4, "Distancia a cada centroide")
                dist = pd.DataFrame({"Segmento": [PERFILES[MAPA[c]]["nombre"].replace("Clientes ", "").capitalize()
                                                  for c in range(K)],
                                     "Distancia (z)": distancias}).sort_values("Distancia (z)")
                st.dataframe(dist, hide_index=True, width="stretch",
                             column_config={"Distancia (z)": st.column_config.ProgressColumn(
                                 format="%.2f", min_value=0, max_value=float(dist.iloc[:, 1].max()))})

    with st.container(key="tarjeta_mapa"):
        etiqueta(5, "Ubicación del cliente")
        grafica(figura_clusters((ingreso, gasto)))
        st.info(f"**Interpretación:** la estrella es el nuevo cliente. K-Means lo asigna al segmento cuyo centroide "
                f"(X) está más cerca después de estandarizar ingreso y gasto: **{info['nombre']}**, a "
                f"{distancias.min():.2f} desviaciones estándar. Si la segunda distancia de la tabla es parecida a "
                f"la primera, el cliente está en la frontera entre dos segmentos.", icon=":material/lightbulb:")

# ------------------------------------------------------------------ clusters
with tab_clusters:
    with st.container(key="tarjeta_clusters"):
        etiqueta(1, "Clusters generados")
        grafica(figura_clusters())
        st.info("**Interpretación:** cada segmento ocupa una zona distinta del plano ingreso–gasto. Arriba a la "
                "derecha están los clientes de **alto valor**, abajo a la derecha los **con potencial**, arriba a "
                "la izquierda los **impulsivos**, abajo a la izquierda los de **bajo valor** y en el centro los "
                "**promedio**. Los grupos están separados por zonas vacías y cada centroide (X) cae en el centro "
                "de su nube, lo que respalda la silueta de 0.555. Pasa el cursor sobre un punto para ver los datos "
                "del cliente.", icon=":material/lightbulb:")

    with st.container(key="tarjeta_resumen"):
        etiqueta(2, "Resumen por segmento")
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

    etiqueta(3, "Perfiles y estrategias")
    cols = st.columns(len(PERFILES))
    for col, (clave_p, info_p) in zip(cols, PERFILES.items()):
        n = int((df["Clave"] == clave_p).sum())
        col.markdown(f"""
        <div class="perfil">
          <div class="barra" style="background:{info_p['color']}"></div>
          <div class="nom">{info_p['nombre']}</div>
          <div class="pct">{n} clientes · {n / len(df):.1%}</div>
          <div class="desc">{DESCRIPCIONES[clave_p]}</div>
          {lista_check(ESTRATEGIAS[clave_p], info_p['color'])}
        </div>""", unsafe_allow_html=True)
    st.write("")
    st.success("**Mayor oportunidad de crecimiento: clientes con potencial.** Tienen el ingreso más alto "
               "(≈ 88 k$) y el gasto más bajo (≈ 17). Si subieran su gasto al nivel promedio, casi lo triplicarían.",
               icon=":material/trending_up:")

# ------------------------------------------------------------------ método del codo
curvas = curvas_k(X_esc)
with tab_codo:
    with st.container(key="tarjeta_codo"):
        etiqueta(1, "Método del codo")
        grafica(figura_curva(curvas["K"], curvas["Inercia"].tolist(), K, "Inercia (WCSS)"))
        tabla = curvas[["K", "Inercia"]].copy()
        tabla["Reducción vs K-1 (%)"] = (-tabla["Inercia"].pct_change() * 100).map(
            lambda v: "—" if pd.isna(v) else f"{v:.1f}")
        c_a, c_b = st.columns([1, 1.4], gap="large")
        c_a.dataframe(tabla, hide_index=True, width="stretch",
                      column_config={"Inercia": st.column_config.NumberColumn(format="%.2f")})
        c_b.info(
            "**Interpretación.** La **inercia** es la suma de distancias al cuadrado de cada cliente a su "
            "centroide: entre más cerca de 0, más compactos son los grupos. Siempre baja al aumentar K, así que se "
            "busca el **codo**, el punto desde el cual agregar clusters ya casi no la reduce.\n\n"
            "De K = 4 a K = 5 la inercia todavía cae un **39.8 %**. Desde K = 6 la reducción es pequeña y la curva "
            "se aplana, así que el codo está en **K = 5**.", icon=":material/lightbulb:")

# ------------------------------------------------------------------ silueta
with tab_silueta:
    with st.container(key="tarjeta_silueta"):
        etiqueta(1, "Coeficiente de silueta")
        sil = curvas.dropna()
        k_sil = int(sil.loc[sil["Silueta"].idxmax(), "K"])
        grafica(figura_curva(sil["K"], sil["Silueta"].tolist(), k_sil, "Silueta"))
        c_a, c_b = st.columns([1, 1.4], gap="large")
        c_a.dataframe(sil[["K", "Silueta", "Davies-Bouldin"]], hide_index=True, width="stretch",
                      column_config={"Silueta": st.column_config.NumberColumn(format="%.4f"),
                                     "Davies-Bouldin": st.column_config.NumberColumn(format="%.3f")})
        c_b.info(
            f"**Interpretación.** La silueta compara la cohesión de cada cliente con su grupo y su separación del "
            f"grupo más cercano. Va de −1 a 1: cerca de **1** significa grupos compactos y bien separados, cerca de "
            f"**0** significa grupos que se superponen y un valor **negativo** significa clientes mal asignados.\n\n"
            f"El máximo está en **K = {k_sil}** (s = {sil['Silueta'].max():.3f}), igual que en el método del codo. "
            f"El índice Davies-Bouldin también es mínimo en ese K.", icon=":material/lightbulb:")

# ------------------------------------------------------------------ datos
with tab_datos:
    with st.container(key="tarjeta_datos"):
        etiqueta(1, "Exploración de los datos")
        c_a, c_b = st.columns(2, gap="large")
        c_a.markdown("**Estadísticas descriptivas**")
        c_a.dataframe(df[["Edad", "Ingreso", "Gasto"]].describe().T.round(2), width="stretch")
        c_b.markdown("**Correlaciones de Pearson**")
        c_b.dataframe(df[["Edad", "Ingreso", "Gasto"]].corr().round(2), width="stretch")
        st.info("**Interpretación:** el dataset no tiene nulos ni duplicados; solo hay 2 valores atípicos de "
                "ingreso (137 k$), que se conservan porque son valores reales. Ingreso y gasto no están "
                "correlacionados (r = 0.01), así que cada uno aporta información distinta, pero forman cinco grupos "
                "claros. La edad tiene una correlación negativa moderada con el gasto (r = −0.33): todos los "
                "clientes con gasto mayor a 60 tienen 40 años o menos.", icon=":material/lightbulb:")
    with st.container(key="tarjeta_tabla"):
        etiqueta(2, "Clientes con su segmento")
        st.dataframe(df[["ID", "Genero", "Edad", "Ingreso", "Gasto", "Perfil"]], hide_index=True,
                     width="stretch", height=360)

# ------------------------------------------------------------------ clasificación masiva
with tab_lote:
    with st.container(key="tarjeta_lote"):
        etiqueta(1, "Clasificar varios clientes desde un CSV")
        st.markdown("Sube un CSV con las columnas `Ingreso` y `Gasto` "
                    "(o `Annual Income (k$)` y `Spending Score (1-100)` como en el dataset original).")
        archivo = st.file_uploader("Archivo CSV", type="csv")
        if archivo is not None:
            lote = pd.read_csv(archivo).rename(columns={"Annual Income (k$)": "Ingreso",
                                                        "Spending Score (1-100)": "Gasto"})
            faltantes = [c for c in FEATURES if c not in lote.columns]
            if faltantes:
                st.error(f"Faltan las columnas: {', '.join(faltantes)}", icon=":material/error:")
            else:
                validos = lote[FEATURES].notna().all(axis=1)
                etiquetas = kmeans.predict(scaler.transform(lote.loc[validos, FEATURES]))
                lote.loc[validos, "Segmento"] = [PERFILES[MAPA[int(c)]]["nombre"] for c in etiquetas]
                if (~validos).any():
                    st.warning(f"{(~validos).sum()} filas tienen valores vacíos y no se clasificaron.",
                               icon=":material/warning:")
                st.dataframe(lote, hide_index=True, width="stretch")
                st.download_button("Descargar resultados", lote.to_csv(index=False).encode("utf-8"),
                                   "clientes_segmentados.csv", "text/csv", icon=":material/download:",
                                   type="primary")

# ------------------------------------------------------------------ informe APA
with tab_informe:
    with st.container(key="tarjeta_informe"):
        etiqueta(1, "Informe en normas APA")
        st.markdown("Documento completo del taller: marco teórico, metodología, respuestas a cada pregunta, "
                    "interpretación de todas las gráficas, estrategias de negocio, despliegue y referencias.")
        c_pdf, c_docx, c_repo = st.columns(3)
        if INFORME_PDF.exists():
            c_pdf.download_button("Descargar PDF", INFORME_PDF.read_bytes(), INFORME_PDF.name, "application/pdf",
                                  icon=":material/picture_as_pdf:", type="primary", width="stretch")
        if INFORME_DOCX.exists():
            c_docx.download_button(
                "Descargar Word", INFORME_DOCX.read_bytes(), INFORME_DOCX.name,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                icon=":material/article:", width="stretch")
        c_repo.link_button("Ver en GitHub", REPOSITORIO, icon=":material/open_in_new:", width="stretch")
    with st.container(key="tarjeta_visor"):
        etiqueta(2, "Vista previa")
        if INFORME_PDF.exists():
            try:
                st.pdf(INFORME_PDF, height=900)
            except Exception:
                st.info("El visor no está disponible en este entorno. Descarga el PDF con el botón de arriba.",
                        icon=":material/info:")
        else:
            st.warning("No se encontró el archivo del informe.", icon=":material/warning:")

# ------------------------------------------------------------------ pie de página
st.markdown(f"""
<div class="pie">
  Valery Nickol Rosero Molina · Ubeimar Lizardo Yepes Portilla<br>
  Inteligencia de Negocios y Minería de Datos · Ingeniería de Software · Universidad Cooperativa de Colombia<br>
  <a href="{REPOSITORIO}" target="_blank" rel="noopener">Repositorio en GitHub</a>
</div>
""", unsafe_allow_html=True)
