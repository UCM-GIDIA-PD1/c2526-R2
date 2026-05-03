"""
minio_service.py

Servicio de acceso a MinIO para la app web.
Descarga rejillas GeoParquet, las convierte a GeoJSON, y cachea en memoria.
"""

import io
import json
import os

import geopandas as gpd
import urllib3
from dotenv import load_dotenv
from minio import Minio

load_dotenv()

# ── Configuración de rejillas disponibles ──────────────────────────────

REJILLAS_CONFIG = {
    "barrios": {
        "nombre": "Barrios",
        "columna_id": "COD_BAR",
        "archivo_minio": "barrios",
    },
    "secciones_censales": {
        "nombre": "Secciones censales",
        "columna_id": "CUSEC",
        "archivo_minio": "secciones censales",
    },
    "hexagonos_1": {
        "nombre": "Hexágonos H3 (grande)",
        "columna_id": "id_hex",
        "archivo_minio": "hexagonos_1",
    },
    "hexagonos_2": {
        "nombre": "Hexágonos H3 (pequeño)",
        "columna_id": "id_hex",
        "archivo_minio": "hexagonos_2",
    },
}

# ── Cliente MinIO (singleton) ──────────────────────────────────────────

_client: Minio | None = None


def _get_client() -> Minio:
    global _client
    if _client is not None:
        return _client

    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    assert endpoint, "Falta MINIO_ENDPOINT en .env"
    assert access_key, "Falta MINIO_ACCESS_KEY en .env"
    assert secret_key, "Falta MINIO_SECRET_KEY en .env"

    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=10.0, read=600.0),
        cert_reqs='CERT_NONE',
        retries=urllib3.Retry(
            total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504]
        ),
    )

    _client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
        http_client=http,
    )
    return _client


# ── Caché en memoria ──────────────────────────────────────────────────

_gdf_cache: dict[str, gpd.GeoDataFrame] = {}
_geojson_cache: dict[str, dict] = {}


def obtener_rejilla(tipo: str) -> gpd.GeoDataFrame:
    """Descarga una rejilla de MinIO y la cachea. Devuelve en EPSG:4326."""
    if tipo in _gdf_cache:
        return _gdf_cache[tipo]

    if tipo not in REJILLAS_CONFIG:
        raise ValueError(f"Rejilla '{tipo}' no existe. Opciones: {list(REJILLAS_CONFIG)}")

    cfg = REJILLAS_CONFIG[tipo]
    client = _get_client()
    bucket = os.getenv("MINIO_BUCKET", "pd1")
    group = os.getenv("MINIO_GROUP_PATH", "grupo2")
    object_path = f"{group}/rejillas/{cfg['archivo_minio']}.parquet"

    response = client.get_object(bucket, object_path)
    buffer = io.BytesIO(response.read())
    response.close()
    response.release_conn()

    gdf = gpd.read_parquet(buffer)

    # Asegurar EPSG:4326 para Leaflet
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    _gdf_cache[tipo] = gdf
    return gdf


def obtener_geojson(tipo: str) -> dict:
    """Devuelve el GeoJSON completo (todas las columnas) de una rejilla."""
    if tipo in _geojson_cache:
        return _geojson_cache[tipo]

    gdf = obtener_rejilla(tipo)
    geojson_str = gdf.to_json()
    geojson = json.loads(geojson_str)

    _geojson_cache[tipo] = geojson
    return geojson


# ── Categorización de columnas ──────────────────────────────────────────

def _categorizar(nombre: str) -> str:
    """Asigna una categoría legible a una columna por su nombre."""
    n = nombre.lower()

    if "precio_m2" in n:
        return "Precios"
    if "media_precio" in n or "media_superficie" in n:
        return "Precios"
    if "num_viviendas" in n or "densidad_viviendas" in n:
        return "Viviendas"
    if "prop_" in n:
        return "Características"
    if any(t in n for t in ("parada", "estacion", "lineas_bus", "lineas_metro")):
        return "Transporte"
    if n.startswith("num_") or n.startswith("densidad_"):
        return "Servicios"
    if any(t in n for t in ("poblacion", "pct_")):
        return "Demografía"
    if "renta" in n:
        return "Economía"
    if "anio" in n or "construccion" in n:
        return "Catastro"
    if n == "area":
        return "General"
    return "Otros"


def _label_legible(nombre: str) -> str:
    """Convierte un nombre de columna en etiqueta legible."""
    label = nombre.replace("_", " ")
    reemplazos = {
        "m2": "m²",
        "pct": "%",
        "num": "Nº",
        "Num": "Nº",
        "anio construccion": "año de construcción",
        "Anio construccion": "Año de construcción",
        "anio": "año",
        "Anio": "Año",
        "construccion": "construcción",
        "prop": "Proporción",
        "Prop": "Proporción",
        "dist min": "Dist. mín.",
    }
    for viejo, nuevo in reemplazos.items():
        label = label.replace(viejo, nuevo)
    
    # Capitalize the first letter but don't lowercase everything else
    # Because .capitalize() lowercases the rest of the string, which might destroy specific casing
    label = label.strip()
    if label:
        label = label[0].upper() + label[1:]
    return label


# Columnas que NO deben mostrarse como variable visualizable
_COLUMNAS_EXCLUIDAS = {
    "geometry", "COD_BAR", "CUSEC", "id_hex", "NOMBRE",
    "total_puntos", "index_right",
}


def listar_columnas_numericas(tipo: str) -> list[dict]:
    """Lista las columnas numéricas de una rejilla, con categoría y label."""
    gdf = obtener_rejilla(tipo)
    columnas = []
    for col in gdf.columns:
        if col in _COLUMNAS_EXCLUIDAS:
            continue
        if gdf[col].dtype in ("float64", "float32", "int64", "int32"):
            columnas.append({
                "nombre": col,
                "categoria": _categorizar(col),
                "label": _label_legible(col),
            })

    # Ordenar por categoría y luego por nombre
    orden_cat = [
        "Precios", "Viviendas", "Características", "Transporte",
        "Servicios", "Demografía", "Economía", "Catastro", "General", "Otros",
    ]
    columnas.sort(key=lambda c: (
        orden_cat.index(c["categoria"]) if c["categoria"] in orden_cat else 99,
        c["nombre"],
    ))
    return columnas


def listar_capas() -> list[dict]:
    """Devuelve el catálogo completo de rejillas con sus columnas."""
    capas = []
    for tipo_id, cfg in REJILLAS_CONFIG.items():
        try:
            columnas = listar_columnas_numericas(tipo_id)
        except Exception:
            columnas = []
        capas.append({
            "id": tipo_id,
            "nombre": cfg["nombre"],
            "columna_id": cfg["columna_id"],
            "columnas": columnas,
        })
    return capas


# ── Datasets Secundarios (Puntos) ──────────────────────────────────────

_secundarios_cache: dict[str, dict] = {}

def listar_secundarios() -> list[str]:
    """Busca en el bucket los archivos de la carpeta secundarios."""
    client = _get_client()
    bucket = os.getenv("MINIO_BUCKET", "pd1")
    group = os.getenv("MINIO_GROUP_PATH", "grupo2")
    prefix = f"{group}/cleaned/secundarios/"
    
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)
    datasets = []
    for obj in objects:
        if obj.object_name.endswith(".parquet"):
            # Extraer solo el nombre base sin extensión
            base = obj.object_name.split("/")[-1].replace(".parquet", "")
            datasets.append(base)
    return datasets

def obtener_geojson_secundario(nombre: str) -> dict:
    """Descarga un parquet secundario (puntos con lat/lon) y devuelve su GeoJSON."""
    if nombre in _secundarios_cache:
        return _secundarios_cache[nombre]

    client = _get_client()
    bucket = os.getenv("MINIO_BUCKET", "pd1")
    group = os.getenv("MINIO_GROUP_PATH", "grupo2")
    object_path = f"{group}/cleaned/secundarios/{nombre}.parquet"

    try:
        response = client.get_object(bucket, object_path)
        buffer = io.BytesIO(response.read())
        response.close()
        response.release_conn()
        
        # Leer como DataFrame normal porque no son GeoParquets
        import pandas as pd
        df = pd.read_parquet(buffer)
        
        # Detectar columnas de coordenadas
        lat_col = 'lat' if 'lat' in df.columns else ('latitud' if 'latitud' in df.columns else None)
        lon_col = 'lon' if 'lon' in df.columns else ('longitud' if 'longitud' in df.columns else None)
        
        if lat_col and lon_col:
            gdf = gpd.GeoDataFrame(
                df, 
                geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
                crs="EPSG:4326"
            )
        else:
            # Fallback por si acaso alguien lo convierte a GeoParquet en el futuro
            buffer.seek(0)
            gdf = gpd.read_parquet(buffer)

        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
            
        geojson_str = gdf.to_json()
        geojson = json.loads(geojson_str)
        _secundarios_cache[nombre] = geojson
        return geojson
    except Exception as e:
        raise ValueError(f"No se pudo cargar el dataset secundario '{nombre}': {e}")
