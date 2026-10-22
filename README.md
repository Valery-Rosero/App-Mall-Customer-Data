# Segmentación de clientes con K-Means

Taller final de Minería de Datos. Segmenta a los 200 clientes del *Mall Customers Dataset* con K-Means según su
**ingreso anual** y su **puntaje de gasto**, y despliega el modelo en una aplicación de Streamlit.

## Resultados

| Métrica | Valor |
|---|---|
| Número de clusters (K) | **5**: el método del codo, la silueta y Davies-Bouldin coinciden |
| Coeficiente de silueta | 0.555 |
| Índice Davies-Bouldin | 0.572 |
| Estabilidad (ARI entre 20 semillas) | 1.000 |

| Segmento | Clientes | Ingreso medio | Gasto medio |
|---|---|---|---|
| Clientes de alto valor | 39 (19.5 %) | 86.5 k$ | 82.1 |
| Clientes con potencial | 35 (17.5 %) | 88.2 k$ | 17.1 |
| Clientes promedio | 81 (40.5 %) | 55.3 k$ | 49.5 |
| Clientes impulsivos | 22 (11.0 %) | 25.7 k$ | 79.4 |
| Clientes de bajo valor | 23 (11.5 %) | 26.3 k$ | 20.9 |

## Estructura

```
├── Taller_KMeans_Segmentacion.ipynb   # Análisis completo (partes 1 a 7) con las respuestas
├── app.py                             # Aplicación Streamlit
├── Mall_Customers.csv                 # Dataset
├── modelo/kmeans_clientes.joblib      # Escalador + K-Means + perfiles (lo genera el notebook)
├── requirements.txt
└── .streamlit/config.toml
```

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## La aplicación

- **Clasificar cliente:** se ingresan el ingreso y el gasto del cliente y la app muestra su segmento, las estrategias
  recomendadas y la distancia a cada centroide.
- **Clusters:** diagrama de dispersión interactivo con los centroides y el resumen de cada perfil.
- **Método del codo** y **Silueta:** curvas de evaluación para K = 1 a 10.
- **Datos:** estadísticas descriptivas y correlaciones.
- **Clasificación masiva:** se sube un CSV y se descarga con el segmento de cada cliente.
