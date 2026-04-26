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

# Destino: datasets preparados para XGBoost en cleaned/datos_primarios
PATH_DESTINO = "cleaned/datos_primarios"

# Columnas que suponen fuga de datos o son meros identificadores
COLS_FUGA = [
    "id", "Nombre", "Direccion", "Tipo_OSM", "geometry", "Precio_m2",
    "Media_precio_venta", "Media_precio_m2_venta",
    "Media_precio_alquiler", "Media_precio_m2_alquiler",
]

# Columnas de alta cardinalidad -> no aportan señal útil tras Kruskal
COLS_ALTA_CARDINALIDAD = ["Calle", "Tipo_Via", "Barrio", "dist_al_edificio"]

# Columnas colineales identificadas mediante matrices Pearson/Spearman/Kendall
# (correlación > 0.8 con otra variable más informativa)
COLS_MULTICOLINEALIDAD = [
    "dist_min_piscinas",          # redundante con cantidad_polideportivos_cerca
    "cantidad_piscinas_cerca",    # redundante con cantidad_polideportivos_cerca
    "estaciones_cerca",           # lineas_distintas_estaciones_cerca es más informativo
    "cantidad_alimentacion_cerca",# dist_min_comercios cubre la señal general
]


# ──────────────────────────────────────────────────────────────────────────────
# Funciones de preparación
# ──────────────────────────────────────────────────────────────────────────────

def limpiar_individuos(df: pd.DataFrame, nombre_mercado: str) -> pd.DataFrame:
    """
    Paso 1 y 2: Elimina filas con nulos críticos, duplicados y outliers de Precio (>P99).
    Preserva la distribución real de los datos sin distorsión por valores extremos.
    """
    print(f"\n[{nombre_mercado.upper()}] Limpieza de individuos...")
    df = df.copy()
    n0 = len(df)

    # 1a. Nulos críticos
    df = df.dropna(subset=["Precio", "lat", "lon", "Superficie"])
    print(f"   -> Nulos críticos eliminados: {n0 - len(df)}")

    # 1b. Duplicados exactos (misma ubicación, precio y superficie)
    n1 = len(df)
    df = df.drop_duplicates(subset=["lat", "lon", "Precio", "Superficie"], keep="first")
    print(f"   -> Duplicados eliminados: {n1 - len(df)}")

    # 2. Outliers de precio: descartar percentil 99
    n2 = len(df)
    p99 = df["Precio"].quantile(0.99)
    df = df[df["Precio"] <= p99]
    print(f"   -> Outliers eliminados (Precio > P99={p99:,.0f} €): {n2 - len(df)}")
    print(f"   -> Individuos restantes: {len(df)}")
    return df


def limpiar_variables(df: pd.DataFrame, nombre_mercado: str) -> pd.DataFrame:
    """
    Pasos 3-5: Elimina columnas por:
      - Fuga de datos e identificadores (Paso 3)
      - Exceso de nulos >40% (Paso 4a)
      - Varianza casi nula: >99% mismo valor (Paso 4b) -> Cocina en ambos mercados,
        Equipamiento en venta
      - Alta cardinalidad post-Kruskal: Calle, Tipo_Via, Barrio, dist_al_edificio (Paso 5)
      - Colinealidad alta >0.8 entre predictores (Paso 6)
      - Cocina en alquiler (tiene varianza, pero el análisis decidió excluirla también)
    """
    print(f"\n[{nombre_mercado.upper()}] Limpieza de variables...")
    df = df.copy()
    n_cols_inicio = df.shape[1]

    # Paso 3: Fuga de datos e identificadores
    cols_fuga_presentes = [c for c in COLS_FUGA if c in df.columns]
    df = df.drop(columns=cols_fuga_presentes)
    print(f"   -> Paso 3 – Fuga/identificadores eliminadas ({len(cols_fuga_presentes)}): {cols_fuga_presentes}")

    # Paso 4a: Exceso de nulos (>40%)
    cols_nulos = df.columns[df.isnull().mean() > 0.40].tolist()
    df = df.drop(columns=cols_nulos)
    if cols_nulos:
        print(f"   -> Paso 4a – Exceso de nulos eliminadas ({len(cols_nulos)}): {cols_nulos}")

    # Paso 4b: Varianza casi nula (>99% mismo valor)
    cols_zero_var = [
        col for col in df.columns
        if df[col].value_counts(normalize=True).iloc[0] > 0.99
    ]
    df = df.drop(columns=cols_zero_var)
    print(f"   -> Paso 4b – Varianza nula eliminadas ({len(cols_zero_var)}): {cols_zero_var}")

    # Paso 5: Alta cardinalidad (Kruskal) y ruido
    cols_card_presentes = [c for c in COLS_ALTA_CARDINALIDAD if c in df.columns]
    df = df.drop(columns=cols_card_presentes)
    if cols_card_presentes:
        print(f"   -> Paso 5 – Alta cardinalidad eliminadas ({len(cols_card_presentes)}): {cols_card_presentes}")

    # Paso 6: Colinealidad alta entre predictores (>0.8, Pearson/Spearman/Kendall)
    cols_multicol_presentes = [c for c in COLS_MULTICOLINEALIDAD if c in df.columns]
    df = df.drop(columns=cols_multicol_presentes)
    if cols_multicol_presentes:
        print(f"   -> Paso 6 – Colineales eliminadas ({len(cols_multicol_presentes)}): {cols_multicol_presentes}")

    # Paso 7 (solo alquiler): Cocina tiene varianza significativa y discrimina
    # según tamaño del piso (88% pisos <30m² equipados vs 34% en >200m²),
    # pero la decisión del análisis fue excluirla igualmente.
    if nombre_mercado.lower() == "alquiler" and "Cocina" in df.columns:
        df = df.drop(columns=["Cocina"])
        print("   -> Paso 7 – 'Cocina' eliminada en alquiler (decisión del análisis estadístico)")

    print(f"   -> Columnas: {n_cols_inicio} -> {df.shape[1]} (eliminadas: {n_cols_inicio - df.shape[1]})")
    print(f"   -> Columnas finales: {df.columns.tolist()}")
    return df


def preparar_dataset_xgboost(df: pd.DataFrame, nombre_mercado: str) -> pd.DataFrame:
    """Pipeline completo de preparación para XGBoost."""
    print(f"\n{'='*70}")
    print(f"PREPARANDO DATASET XGBOOST: {nombre_mercado.upper()}")
    print(f"{'='*70}")
    print(f"Dimensiones iniciales: {df.shape}")

    df = limpiar_individuos(df, nombre_mercado)
    df = limpiar_variables(df, nombre_mercado)

    print(f"\n[OK] Dataset final: {df.shape[0]} individuos x {df.shape[1]} variables")
    return df.reset_index(drop=True)


def mostrar_resumen(datasets: Dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 80)
    print("RESUMEN FINAL DE DATASETS GENERADOS")
    print("=" * 80)
    for nombre, df in datasets.items():
        print(f"  {nombre:<30} -> filas={df.shape[0]:>6} | columnas={df.shape[1]:>4}")
    print("=" * 80)
    print(f"\nSubidos a MinIO en: {PATH_DESTINO}/")


def main() -> None:
    # Cliente robusto con timeout extendido para archivos grandes
    client_descarga = _crear_cliente_robusto()
    client_subida   = crear_cliente_minio()

    print("Descargando datasets enriquecidos desde MinIO (rejillas/)...")
    df_venta    = _descargar_geoparquet(client_descarga, PATH_FUENTE, OBJ_VENTA)
    df_alquiler = _descargar_geoparquet(client_descarga, PATH_FUENTE, OBJ_ALQUILER)

    print(f"  -> venta:    {df_venta.shape}")
    print(f"  -> alquiler: {df_alquiler.shape}")

    # Preparación
    df_venta_xgboost    = preparar_dataset_xgboost(df_venta,    "venta")
    df_alquiler_xgboost = preparar_dataset_xgboost(df_alquiler, "alquiler")

    datasets = {
        "df_venta_xgboost":    df_venta_xgboost,
        "df_alquiler_xgboost": df_alquiler_xgboost,
    }

    # Subida a MinIO
    print("\nSubiendo datasets a MinIO...")
    for nombre, df in datasets.items():
        subir_minio(
            df=df,
            client=client_subida,
            path=PATH_DESTINO,
            minio_object=f"{nombre}.parquet",
        )
        print(f"  [OK] {nombre}.parquet subido a {PATH_DESTINO}/")

    mostrar_resumen(datasets)


if __name__ == "__main__":
    main()