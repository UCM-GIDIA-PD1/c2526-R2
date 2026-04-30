"""
enrichment.py

Servicio de enriquecimiento de datos para predicciones.
Replica el pipeline de geocodificación y cálculo de distancias
de los scripts de transformación (B_viviendas_madrid.py),
pero ejecutado en tiempo real para una sola vivienda.
"""

import io
import os
import re
import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from geopy.geocoders import Nominatim
from minio import Minio
from dotenv import load_dotenv
import urllib3

logger = logging.getLogger(__name__)

# ─── Constants (mirrors src/utils/config.py) ───────────────────────────

PAIS = "Spain"
CIUDAD = "Madrid"

COMPONENTES_TRANSPORTE = [
    {"tipo": "bus", "fichero": "paradas_bus.parquet", "calculo": "paradas"},
    {"tipo": "metro", "fichero": "estaciones_metro.parquet", "calculo": "estaciones"},
]

# Categories of secondary data stored in cleaned/secundarios
CATEGORIAS_SECUNDARIOS = [
    "alimentacion", "bibliotecas", "bomberos", "cementerios",
    "centros_dia", "centros_educativos", "centros_mayores",
    "centros_sociales", "comercios", "comisarias", "hospitales",
    "iglesias", "negativos", "parques", "parques_bomberos",
    "polideportivos", "puntos_limpios", "servicios_sociales",
    "universidades",
]

DISTRITOS = [
    "CENTRO", "ARGANZUELA", "RETIRO", "SALAMANCA", "CHAMARTIN",
    "TETUAN", "CHAMBERI", "FUENCARRAL-EL PARDO", "MONCLOA-ARAVACA",
    "LATINA", "CARABANCHEL", "USERA", "PUENTE DE VALLECAS",
    "MORATALAZ", "CIUDAD LINEAL", "HORTALEZA", "VILLAVERDE",
    "VILLA DE VALLECAS", "VICALVARO", "SAN BLAS-CANILLEJAS", "BARAJAS",
]

RADIO_METROS = 500


# ─── MinIO client ───────────────────────────────────────────────────────

def _crear_cliente_minio() -> Minio:
    """Creates a MinIO client from .env vars (same logic as src/utils).

    Note: _region_map is pre-seeded to avoid the automatic region-detection
    HEAD request, which fails because the UCM proxy returns an HTML login page
    instead of the expected S3 XML response, causing a ParseError.
    """
    load_dotenv()
    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    bucket = os.getenv("MINIO_BUCKET", "pd1")

    assert endpoint, "Falta MINIO_ENDPOINT en el entorno/.env"
    assert access_key, "Falta MINIO_ACCESS_KEY en el entorno/.env"
    assert secret_key, "Falta MINIO_SECRET_KEY en el entorno/.env"

    http_client = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=10.0, read=120.0),
        retries=urllib3.Retry(
            total=3, backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        ),
    )
    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
        http_client=http_client,
    )
    # Pre-seed region to skip auto-detection (UCM proxy returns HTML, not XML)
    client._region_map[bucket] = "us-east-1"
    return client


def _minio_path() -> tuple[str, str]:
    """Returns (bucket, group_path) from env."""
    load_dotenv()
    bucket = os.getenv("MINIO_BUCKET")
    group_path = os.getenv("MINIO_GROUP_PATH")
    assert bucket, "Falta MINIO_BUCKET"
    assert group_path, "Falta MINIO_GROUP_PATH"
    return bucket, group_path


def _bajar_df(client: Minio, path: str, obj: str) -> pd.DataFrame:
    bucket, gp = _minio_path()
    full = f"{gp}/{path}/{obj}"
    resp = client.get_object(bucket, full)
    data = io.BytesIO(resp.read())
    df = pd.read_parquet(data)
    resp.close()
    resp.release_conn()
    return df


def _bajar_gdf(client: Minio, path: str, nombre: str) -> gpd.GeoDataFrame:
    bucket, gp = _minio_path()
    full = f"{gp}/{path}/{nombre}.parquet"
    resp = client.get_object(bucket, full)
    buf = io.BytesIO(resp.read())
    gdf = gpd.read_parquet(buf)
    resp.close()
    resp.release_conn()
    return gdf


# ─── Cached reference data (loaded once, reused) ───────────────────────

class ReferenceDataStore:
    """Singleton that lazy-loads all reference datasets from MinIO."""

    def __init__(self):
        self._loaded = False
        self._secundarios: dict[str, pd.DataFrame] = {}
        self._transporte: dict[str, pd.DataFrame] = {}
        self._catastro: Optional[gpd.GeoDataFrame] = None
        self._padron: Optional[gpd.GeoDataFrame] = None
        # Pre-built KD-trees for fast spatial queries
        self._trees_secundarios: dict[str, tuple[cKDTree, np.ndarray, np.ndarray]] = {}
        self._trees_transporte: dict[str, tuple[cKDTree, pd.DataFrame]] = {}

    def load(self):
        if self._loaded:
            return
        logger.info("Cargando datos de referencia desde MinIO...")
        client = _crear_cliente_minio()

        # --- Secondary data (amenities) ---
        for cat in CATEGORIAS_SECUNDARIOS:
            try:
                df = _bajar_df(client, "cleaned/secundarios", f"{cat}.parquet")
                self._secundarios[cat] = df
                # Build KD-tree in EPSG:25830
                gdf_poi = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                    crs="EPSG:4326",
                ).to_crs("EPSG:25830")
                coords = np.column_stack([gdf_poi.geometry.x, gdf_poi.geometry.y])
                tree = cKDTree(coords)
                self._trees_secundarios[cat] = (tree, coords, gdf_poi.geometry.x.values)
                logger.info(f"  Cargado secundario: {cat} ({len(df)} registros)")
            except Exception as e:
                logger.warning(f"  No se pudo cargar secundario '{cat}': {e}")

        # --- Transport ---
        for comp in COMPONENTES_TRANSPORTE:
            try:
                df = _bajar_df(client, "cleaned/transporte", comp["fichero"])
                nombre = comp["calculo"]
                self._transporte[nombre] = df
                gdf_t = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                    crs="EPSG:4326",
                ).to_crs("EPSG:25830")
                coords = np.column_stack([gdf_t.geometry.x, gdf_t.geometry.y])
                tree = cKDTree(coords)
                self._trees_transporte[nombre] = (tree, df)
                logger.info(f"  Cargado transporte: {nombre} ({len(df)} registros)")
            except Exception as e:
                logger.warning(f"  No se pudo cargar transporte '{comp['calculo']}': {e}")

        # --- Catastro ---
        try:
            self._catastro = _bajar_gdf(client, "cleaned/catastro", "anio_construccion")
            logger.info(f"  Cargado catastro ({len(self._catastro)} polígonos)")
        except Exception as e:
            logger.warning(f"  No se pudo cargar catastro: {e}")

        # --- Padrón / Secciones censales ---
        try:
            self._padron = _bajar_gdf(client, "rejillas", "secciones censales")
            logger.info(f"  Cargado padrón ({len(self._padron)} secciones)")
        except Exception as e:
            logger.warning(f"  No se pudo cargar padrón: {e}")

        self._loaded = True
        logger.info("Datos de referencia cargados correctamente.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Global singleton
_store = ReferenceDataStore()


def get_store() -> ReferenceDataStore:
    """Returns the singleton reference data store, loading if needed."""
    if not _store.is_loaded:
        _store.load()
    return _store


# ─── Geocoding ──────────────────────────────────────────────────────────

def _limpia_direccion(direccion: str) -> str:
    """Same address cleaning as src/2_Limpieza/A_viviendas_modificacion.py."""
    dir_limpia = direccion.upper()
    abreviaturas = {
        r'\bPS\b': 'PASEO',
        r'\bC/\b': 'CALLE',
        r'\bC\.\b': 'CALLE',
        r'\bCl\b': 'CALLE',
        r'\bC\b': 'CALLE',
        r'\bAV\b': 'AVENIDA',
        r'\bAVDA\b': 'AVENIDA',
        r'\bPL\b': 'PLAZA',
        r'\bCTRA\b': 'CARRETERA',
    }
    for abr, sub in abreviaturas.items():
        dir_limpia = re.sub(abr, sub, dir_limpia)
    dir_limpia = re.sub(r'\s*S/N-?\s*', ' ', dir_limpia)
    dir_limpia = dir_limpia.split(' - ')[0]
    partes = dir_limpia.split(',')
    if len(partes) > 2:
        dir_limpia = f"{partes[0].strip()}, {partes[1].strip()}"
    dir_limpia = re.sub(r',?\s*\d+[ºª].*$', '', dir_limpia)
    dir_limpia = re.sub(r',?\s*\d+-\d+\s*$', '', dir_limpia)
    dir_limpia = re.sub(r'\s+', ' ', dir_limpia).strip()
    dir_limpia = re.sub(r',$', '', dir_limpia)
    return f"{dir_limpia}, {CIUDAD}, {PAIS}"


def geocode_address(direccion: str) -> tuple[Optional[float], Optional[float]]:
    """Geocode a street address using Nominatim (same as cleaning pipeline)."""
    dir_limpia = _limpia_direccion(direccion)
    try:
        geolocator = Nominatim(user_agent="maiday_web_v1")
        loc = geolocator.geocode(dir_limpia, addressdetails=True)
        if loc:
            return loc.latitude, loc.longitude
    except Exception as e:
        logger.warning(f"Error en geocodificación Nominatim: {e}")

    # Fallback: try Google Maps if API key available
    try:
        api_key = os.getenv("API_GOOGLE")
        if api_key:
            import googlemaps
            gmaps = googlemaps.Client(key=api_key)
            resultado = gmaps.geocode(dir_limpia)
            if resultado:
                loc = resultado[0]['geometry']['location']
                return loc['lat'], loc['lng']
    except Exception as e:
        logger.warning(f"Error en geocodificación Google: {e}")

    return None, None


# ─── Spatial enrichment functions ───────────────────────────────────────

def _point_to_25830(lat: float, lon: float) -> tuple[float, float]:
    """Convert a single WGS84 point to EPSG:25830 coordinates."""
    gdf = gpd.GeoDataFrame(
        {"lat": [lat], "lon": [lon]},
        geometry=gpd.points_from_xy([lon], [lat]),
        crs="EPSG:4326",
    ).to_crs("EPSG:25830")
    return float(gdf.geometry.x.iloc[0]), float(gdf.geometry.y.iloc[0])


def compute_secondary_features(lat: float, lon: float, store: ReferenceDataStore) -> dict:
    """Compute dist_min and cantidad_cerca for all secondary POI categories."""
    x, y = _point_to_25830(lat, lon)
    result = {}

    for cat, (tree, coords, _) in store._trees_secundarios.items():
        # Distance to nearest
        dist, _ = tree.query([x, y], k=1)
        result[f"dist_min_{cat}"] = round(float(dist), 1)

        # Count within radius
        indices = tree.query_ball_point([x, y], r=RADIO_METROS)
        result[f"cantidad_{cat}_cerca"] = len(indices)

    return result


def compute_transport_features(lat: float, lon: float, store: ReferenceDataStore) -> dict:
    """Compute transport features (same logic as meter_datos_transporte)."""
    x, y = _point_to_25830(lat, lon)
    result = {}

    for nombre, (tree, df) in store._trees_transporte.items():
        # Distance to nearest stop/station
        dist, _ = tree.query([x, y], k=1)
        result[f"dist_min_{nombre}"] = round(float(dist), 1)

        # Stops within radius
        indices = tree.query_ball_point([x, y], r=RADIO_METROS)
        result[f"{nombre}_cerca"] = len(indices)

        # Distinct lines within radius
        lineas_array = df["lineas"].values
        lineas_unicas = set()
        for idx in indices:
            item = lineas_array[idx]
            if isinstance(item, (list, tuple)):
                lineas_unicas.update(item)
            elif isinstance(item, str):
                lineas_unicas.update([s.strip() for s in item.split(",") if s.strip()])
        result[f"lineas_distintas_{nombre}_cerca"] = len(lineas_unicas)

    return result


def compute_catastro_features(lat: float, lon: float, store: ReferenceDataStore) -> dict:
    """Get año de construcción from catastro via spatial nearest join."""
    result = {"anio_construccion": 1980.0}  # default

    if store._catastro is not None:
        try:
            gdf_point = gpd.GeoDataFrame(
                {"lat": [lat], "lon": [lon]},
                geometry=gpd.points_from_xy([lon], [lat]),
                crs="EPSG:4326",
            )
            catastro = store._catastro.copy()
            if gdf_point.crs != catastro.crs:
                gdf_point = gdf_point.to_crs(catastro.crs)

            cruce = gpd.sjoin_nearest(
                gdf_point,
                catastro[["geometry", "anio_construccion"]],
                how="left",
                max_distance=30,
                distance_col="dist_al_edificio",
            )
            if not cruce.empty and pd.notna(cruce.iloc[0].get("anio_construccion")):
                result["anio_construccion"] = float(cruce.iloc[0]["anio_construccion"])
        except Exception as e:
            logger.warning(f"Error calculando catastro: {e}")

    return result


def compute_padron_features(lat: float, lon: float, store: ReferenceDataStore) -> dict:
    """Get demographic data from padrón via spatial join."""
    result = {
        "poblacion_total": 0.0,
        "pct_extranjeros": 0.0,
        "pct_mayores_65": 0.0,
        "pct_jovenes_30": 0.0,
    }

    if store._padron is not None:
        try:
            gdf_point = gpd.GeoDataFrame(
                {"lat": [lat], "lon": [lon]},
                geometry=gpd.points_from_xy([lon], [lat]),
                crs="EPSG:4326",
            )
            padron = store._padron.copy()
            if gdf_point.crs != padron.crs:
                gdf_point = gdf_point.to_crs(padron.crs)

            cols_transfer = [
                c for c in ["geometry", "poblacion_total", "pct_extranjeros",
                             "pct_mayores_65", "pct_jovenes_30"]
                if c in padron.columns
            ]
            cruce = gpd.sjoin(
                gdf_point,
                padron[cols_transfer],
                how="left",
                predicate="within",
            )
            if not cruce.empty:
                row = cruce.iloc[0]
                for key in ["poblacion_total", "pct_extranjeros", "pct_mayores_65", "pct_jovenes_30"]:
                    if key in row and pd.notna(row[key]):
                        result[key] = float(row[key])
        except Exception as e:
            logger.warning(f"Error calculando padrón: {e}")

    return result


# ─── Main enrichment function ──────────────────────────────────────────

def enrich_property(direccion: str, basic_data: dict) -> dict:
    """
    Takes user-provided address + basic property data and enriches it
    with all computed spatial features needed by the model.

    Returns a complete feature dict ready for model prediction.
    """
    store = get_store()

    # Step 1: Geocode the address
    lat, lon = geocode_address(direccion)
    if lat is None or lon is None:
        raise ValueError(
            f"No se pudo geocodificar la dirección: '{direccion}'. "
            "Comprueba que es una dirección válida en Madrid."
        )

    # Validate coordinates are within Madrid bounds
    if not (40.28 <= lat <= 40.65 and -3.83 <= lon <= -3.48):
        raise ValueError(
            f"Las coordenadas ({lat}, {lon}) están fuera de Madrid. "
            "Introduce una dirección dentro de la ciudad."
        )

    # Step 2: Compute all derived features
    features = dict(basic_data)  # Start with user-provided data
    features["lat"] = lat
    features["lon"] = lon

    # Secondary POIs (alimentacion, bibliotecas, etc.)
    features.update(compute_secondary_features(lat, lon, store))

    # Transport (bus stops, metro stations)
    features.update(compute_transport_features(lat, lon, store))

    # Catastro (year of construction) — only if not provided by user
    if "anio_construccion" not in features or not features["anio_construccion"]:
        catastro = compute_catastro_features(lat, lon, store)
        features.update(catastro)

    # Demographics (padrón)
    features.update(compute_padron_features(lat, lon, store))

    return features
