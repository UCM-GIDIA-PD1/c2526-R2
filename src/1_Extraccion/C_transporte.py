"""
C_transporte.py

Extracción de datos de transporte público de Madrid (paradas de bus y estaciones de metro).

Fuente: ArcGIS REST API del Consorcio Regional de Transportes de Madrid (CRTM).
    - Bus:   servicio M6_Red, capa 0. Devuelve JSON (f=json).
    - Metro: servicio Lineas_Metro, múltiples capas de estaciones (una por línea).
             Se consultan todas y se deduplicen por nombre de estación.

Campos extraídos: DENOMINACION, coordenadas (geometry x,y).

Pipeline:
    1. Descarga el JSON desde la API de ArcGIS
    2. Parsea features[].attributes y features[].geometry
    3. Lo convierte a Parquet en memoria
    4. Lo sube a MinIO en grupo2/raw/
"""

import os
import io
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from dotenv import load_dotenv
from funciones_minio import crear_cliente_minio, minio_subir_memoria

# ── Bus ─────────────────────────────────────────────────────────────────────
# URL del servicio M6_Red (paradas de bus) en ArcGIS
URL_BUS = (
    "https://services5.arcgis.com/UxADft6QPcvFyDU1/arcgis/rest/services/"
    "M6_Red/FeatureServer/0/query?where=1%3D1"
    "&outFields=DENOMINACION,X,Y,GRADOACCESIBILIDAD&outSR=4326&f=json"
)

# ── Metro ───────────────────────────────────────────────────────────────────
# Base del servicio Lineas_Metro en ArcGIS
URL_METRO_BASE = (
    "https://services5.arcgis.com/UxADft6QPcvFyDU1/arcgis/rest/services/"
    "Lineas_Metro/FeatureServer"
)

# IDs de las capas de ESTACION (sentido 1 = S1, para cada línea de metro)
# L1(2), L2(11), L3(20), L4(29), L5(38), L6(47), L7a(56), L7b(59),
# L8(71), L9A(80), L9B(83), L10a(95), L10b(98), L11(110), L12(119), LR(128)
METRO_LAYER_IDS = [2, 11, 20, 29, 38, 47, 56, 59, 71, 80, 83, 95, 98, 110, 119, 128]

# Rutas de destino en MinIO
OBJECTS = {
    "bus": "grupo2/raw/paradas_bus.parquet",
    "metro": "grupo2/raw/estaciones_metro.parquet",
}


def _parsear_arcgis_json(data: dict) -> gpd.GeoDataFrame:
    """Convierte la respuesta JSON de ArcGIS (f=json) a GeoDataFrame.

    El formato ArcGIS JSON tiene la estructura:
        { "features": [ { "attributes": {...}, "geometry": {"x": ..., "y": ...} }, ... ] }

    Args:
        data: Diccionario con la respuesta JSON de ArcGIS.

    Returns:
        GeoDataFrame con los atributos y geometría de cada feature.
    """
    features = data.get("features", [])
    rows = []
    geometries = []
    for feat in features:
        rows.append(feat.get("attributes", {}))
        geom = feat.get("geometry", {})
        geometries.append(Point(geom.get("x", 0), geom.get("y", 0)))

    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    return gdf


def descargar_bus() -> io.BytesIO:
    """Descarga paradas de bus desde ArcGIS y las convierte a Parquet en memoria.

    Returns:
        Buffer en memoria con el contenido en formato Parquet.
    """
    print("  Descargando paradas de bus...")
    response = requests.get(URL_BUS)
    response.raise_for_status()

    gdf = _parsear_arcgis_json(response.json())
    print(f"  Paradas de bus obtenidas: {len(gdf)}")

    buffer = io.BytesIO()
    gdf.to_parquet(buffer, index=False)
    buffer.seek(0)
    return buffer


def descargar_metro() -> io.BytesIO:
    """Descarga estaciones de metro de todas las líneas y las combina en un Parquet.

    Consulta las 16 capas de estaciones (una por línea) del servicio Lineas_Metro,
    las combina en un solo GeoDataFrame y elimina duplicados por nombre de estación
    (las estaciones de transbordo aparecen en varias líneas).

    Returns:
        Buffer en memoria con el contenido en formato Parquet.
    """
    print("  Descargando estaciones de metro (todas las líneas)...")
    todos = []

    for layer_id in METRO_LAYER_IDS:
        url = (
            f"{URL_METRO_BASE}/{layer_id}/query?where=1%3D1"
            f"&outFields=DENOMINACION&outSR=4326&f=json"
        )
        response = requests.get(url)
        response.raise_for_status()
        gdf_linea = _parsear_arcgis_json(response.json())
        if len(gdf_linea) > 0:
            todos.append(gdf_linea)

    # Combinar todas las líneas y eliminar estaciones duplicadas (transbordos)
    gdf = pd.concat(todos, ignore_index=True)
    gdf = gdf.drop_duplicates(subset="DENOMINACION").reset_index(drop=True)
    print(f"  Estaciones de metro únicas: {len(gdf)}")

    buffer = io.BytesIO()
    gdf.to_parquet(buffer, index=False)
    buffer.seek(0)
    return buffer


if __name__ == "__main__":
    load_dotenv()
    client = crear_cliente_minio()

    # Bus
    print("Procesando: bus")
    try:
        buffer = descargar_bus()
        minio_subir_memoria(client, buffer, OBJECTS["bus"])
        print(f"  bus subido a MinIO -> {OBJECTS['bus']}")
    except Exception as e:
        print(f"  ERROR en bus: {e}")

    # Metro
    print("Procesando: metro")
    try:
        buffer = descargar_metro()
        minio_subir_memoria(client, buffer, OBJECTS["metro"])
        print(f"  metro subido a MinIO -> {OBJECTS['metro']}")
    except Exception as e:
        print(f"  ERROR en metro: {e}")

    print("Extracción de transporte finalizada.")
