import io
import os
import sys
from pathlib import Path
from typing import Dict

import geopandas as gpd
import pandas as pd
import urllib3
from dotenv import load_dotenv
from minio import Minio

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de imports del proyecto
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
print(f"ROOT detectado: {ROOT}")

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    print(f"Añadido al path: {ROOT}")

from utils.funciones_minio import crear_cliente_minio, subir_minio


def _crear_cliente_robusto() -> Minio:
    """Cliente MinIO con timeout de 30 min y 10 reintentos para descargas grandes."""
    load_dotenv()
    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=30.0, read=1800.0),
        retries=urllib3.Retry(total=10, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504]),
    )
    return Minio(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=True,
        http_client=http,
    )


def _descargar_geoparquet(client: Minio, path: str, nombre: str) -> pd.DataFrame:
    """Descarga un GeoParquet de MinIO y devuelve un DataFrame plano (sin geometría)."""
    load_dotenv()
    bucket = os.getenv("MINIO_BUCKET")
    group  = os.getenv("MINIO_GROUP_PATH")
    obj_name = f"{group}/{path}/{nombre}.parquet"
    print(f"   Descargando {obj_name} ...")
    respuesta = client.get_object(bucket_name=bucket, object_name=obj_name)
    buffer = io.BytesIO(respuesta.read())
    respuesta.close()
    respuesta.release_conn()
    # Leer como GeoDataFrame y convertir a DataFrame plano
    gdf = gpd.read_parquet(buffer)
    return pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))


# ──────────────────────────────────────────────────────────────────────────────
# Parámetros
# ──────────────────────────────────────────────────────────────────────────────

# Fuente: datasets enriquecidos con features geoespaciales (rejillas/)
PATH_FUENTE = "rejillas"
OBJ_VENTA    = "viviendas_venta"
OBJ_ALQUILER = "viviendas_alquiler"

# Destino: Ajustado a dataset/precios/
PATH_DESTINO_BASE = "dataset_ml/precios"

# Columnas que suponen fuga de datos o son meros identificadores
COLS_FUGA = [
    "id", "Nombre", "Direccion", "Tipo_OSM", "geometry", "Precio_m2",
    "Media_precio_venta", "Media_precio_m2_venta",
    "Media_precio_alquiler", "Media_precio_m2_alquiler",
]

COLS_ALTA_CARDINALIDAD = ["Calle", "Tipo_Via", "Barrio", "dist_al_edificio"]

COLS_MULTICOLINEALIDAD = [
    "dist_min_piscinas",          
    "cantidad_piscinas_cerca",    
    "estaciones_cerca",           
    "cantidad_alimentacion_cerca",
]


# ──────────────────────────────────────────────────────────────────────────────
# Funciones de preparación
# ──────────────────────────────────────────────────────────────────────────────

def limpiar_individuos(df: pd.DataFrame, nombre_mercado: str) -> pd.DataFrame:
    print(f"\n[{nombre_mercado.upper()}] Limpieza de individuos...")
    df = df.copy()
    n0 = len(df)
    df = df.dropna(subset=["Precio", "lat", "lon", "Superficie"])
    print(f"   -> Nulos críticos eliminados: {n0 - len(df)}")

    n1 = len(df)
    df = df.drop_duplicates(subset=["lat", "lon", "Precio", "Superficie"], keep="first")
    print(f"   -> Duplicados eliminados: {n1 - len(df)}")

    n2 = len(df)
    p99 = df["Precio"].quantile(0.99)
    df = df[df["Precio"] <= p99]
    print(f"   -> Outliers eliminados (Precio > P99={p99:,.0f} €): {n2 - len(df)}")
    return df


def limpiar_variables(df: pd.DataFrame, nombre_mercado: str) -> pd.DataFrame:
    print(f"\n[{nombre_mercado.upper()}] Limpieza de variables...")
    df = df.copy()
    n_cols_inicio = df.shape[1]

    cols_fuga_presentes = [c for c in COLS_FUGA if c in df.columns]
    df = df.drop(columns=cols_fuga_presentes)

    cols_nulos = df.columns[df.isnull().mean() > 0.40].tolist()
    df = df.drop(columns=cols_nulos)

    cols_zero_var = [
        col for col in df.columns
        if df[col].value_counts(normalize=True).iloc[0] > 0.99
    ]
    df = df.drop(columns=cols_zero_var)

    cols_card_presentes = [c for c in COLS_ALTA_CARDINALIDAD if c in df.columns]
    df = df.drop(columns=cols_card_presentes)

    cols_multicol_presentes = [c for c in COLS_MULTICOLINEALIDAD if c in df.columns]
    df = df.drop(columns=cols_multicol_presentes)

    if nombre_mercado.lower() == "alquiler" and "Cocina" in df.columns:
        df = df.drop(columns=["Cocina"])

    print(f"   -> Columnas finales: {df.shape[1]} (eliminadas: {n_cols_inicio - df.shape[1]})")
    return df


def preparar_dataset_xgboost(df: pd.DataFrame, nombre_mercado: str) -> pd.DataFrame:
    print(f"\n{'='*70}\nPREPARANDO DATASET XGBOOST: {nombre_mercado.upper()}\n{'='*70}")
    df = limpiar_individuos(df, nombre_mercado)
    df = limpiar_variables(df, nombre_mercado)
    return df.reset_index(drop=True)


def main() -> None:
    client_descarga = _crear_cliente_robusto()
    client_subida   = crear_cliente_minio()

    print("Descargando datasets desde MinIO...")
    df_venta_raw    = _descargar_geoparquet(client_descarga, PATH_FUENTE, OBJ_VENTA)
    df_alquiler_raw = _descargar_geoparquet(client_descarga, PATH_FUENTE, OBJ_ALQUILER)

    # Procesamiento
    df_venta_proc    = preparar_dataset_xgboost(df_venta_raw, "venta")
    df_alquiler_proc = preparar_dataset_xgboost(df_alquiler_raw, "alquiler")

    # Estructura de datos para iterar: {nombre_archivo: (dataframe, subcarpeta)}
    datasets = {
        "df_venta_xgboost": (df_venta_proc, "ventas"),
        "df_alquiler_xgboost": (df_alquiler_proc, "alquiler")
    }

    print("\nSubiendo datasets a MinIO...")
    for nombre, (df, mercado) in datasets.items():
        # La ruta final será: dataset/precios/venta o dataset/precios/alquiler
        ruta_final = f"{PATH_DESTINO_BASE}/{mercado}"
        
        subir_minio(
            df=df,
            client=client_subida,
            path=ruta_final,
            minio_object=f"{nombre}.parquet",
        )
        print(f"   [OK] {nombre}.parquet subido a {ruta_final}/")

    print("PROCESO FINALIZADO")

if __name__ == "__main__":
    main()