import io
import os

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from minio import Minio

from utils.funciones_minio import crear_cliente_minio, subir_minio


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


# Parámetros
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

# Columnas con alta cardinalidad (no aportan valor predictivo)
COLS_ALTA_CARDINALIDAD = ["Calle", "Tipo_Via", "Barrio", "dist_al_edificio"]

# Columnas con multicolinealidad (redundantes)
COLS_MULTICOLINEALIDAD = [
    "dist_min_piscinas",          
    "cantidad_piscinas_cerca",    
    "estaciones_cerca",           
    "cantidad_alimentacion_cerca",
]


# Funciones de preparación (reflejando el notebook analisis_estadistico_2.ipynb)
def limpiar_individuos(df: pd.DataFrame, nombre_mercado: str) -> pd.DataFrame:
    """Paso 1: Eliminar nulos críticos, duplicados y outliers (P99)."""
    print(f"\n[{nombre_mercado.upper()}] Limpieza de individuos...")
    df = df.copy()
    
    # 1. Nulos críticos
    n0 = len(df)
    df = df.dropna(subset=["Precio", "lat", "lon", "Superficie"])
    nulos_eli = n0 - len(df)
    print(f"   -> Nulos críticos eliminados: {nulos_eli}")

    # 2. Duplicados
    n1 = len(df)
    df = df.drop_duplicates(subset=["lat", "lon", "Precio", "Superficie"], keep="first")
    dup_eli = n1 - len(df)
    print(f"   -> Duplicados eliminados: {dup_eli}")

    # 3. Outliers (P99)
    n2 = len(df)
    p99 = df["Precio"].quantile(0.99)
    df = df[df["Precio"] <= p99]
    out_eli = n2 - len(df)
    print(f"   -> Outliers eliminados (Precio > P99={p99:,.0f} €): {out_eli}")
    
    print(f"   -> Filas finales: {len(df)}")
    return df


def limpiar_variables(df: pd.DataFrame, nombre_mercado: str) -> pd.DataFrame:
    """Paso 2: Eliminar variables con fuga de datos, nulos, varianza cero, 
    alta cardinalidad y multicolinealidad."""
    print(f"\n[{nombre_mercado.upper()}] Limpieza de variables...")
    df = df.copy()
    n_cols_inicio = df.shape[1]
    print(f"   -> Columnas iniciales: {n_cols_inicio}")

    # 1. Fuga de datos e identificadores
    cols_fuga_presentes = [c for c in COLS_FUGA if c in df.columns]
    df = df.drop(columns=cols_fuga_presentes)
    print(f"   -> Eliminadas {len(cols_fuga_presentes)} columnas por fuga/identificadores: {cols_fuga_presentes}")

    # 2. Columnas con >40% de nulos
    cols_nulos = df.columns[df.isnull().mean() > 0.40].tolist()
    df = df.drop(columns=cols_nulos)
    print(f"   -> Eliminadas {len(cols_nulos)} columnas por exceso de nulos (>40%): {cols_nulos}")

    # 3. Varianza cero (>99% mismo valor)
    cols_zero_var = [
        col for col in df.columns
        if df[col].value_counts(normalize=True).iloc[0] > 0.99
    ]
    df = df.drop(columns=cols_zero_var)
    print(f"   -> Eliminadas {len(cols_zero_var)} columnas por varianza cero: {cols_zero_var}")

    # 4. Alta cardinalidad
    cols_card_presentes = [c for c in COLS_ALTA_CARDINALIDAD if c in df.columns]
    df = df.drop(columns=cols_card_presentes)
    print(f"   -> Eliminadas {len(cols_card_presentes)} columnas por alta cardinalidad: {cols_card_presentes}")

    # 5. Multicolinealidad
    cols_multicol_presentes = [c for c in COLS_MULTICOLINEALIDAD if c in df.columns]
    df = df.drop(columns=cols_multicol_presentes)
    print(f"   -> Eliminadas {len(cols_multicol_presentes)} columnas por multicolinealidad: {cols_multicol_presentes}")

    # 6. Para alquiler, eliminar Cocina (analizamos su distribución por rango de superficie)
    if nombre_mercado.lower() == "alquiler" and "Cocina" in df.columns:
        df = df.drop(columns=["Cocina"])
        print(f"   -> Eliminada columna 'Cocina' (específico para mercado de alquiler)")

    n_cols_final = df.shape[1]
    cols_eliminadas = n_cols_inicio - n_cols_final
    print(f"   -> Columnas finales: {n_cols_final} (eliminadas en total: {cols_eliminadas})")
    
    return df


def preparar_dataset_completo(df: pd.DataFrame, nombre_mercado: str) -> pd.DataFrame:
    """Pipeline completo: limpieza de individuos + limpieza de variables."""
    print(f"PREPARANDO DATASET PARA XGBOOST: {nombre_mercado.upper()}")
    
    df = limpiar_individuos(df, nombre_mercado)
    df = limpiar_variables(df, nombre_mercado)
    
    # Resumen final
    print(f"\n[{nombre_mercado.upper()}] Resumen final:")
    print(f"   -> Dimensiones: {df.shape}")
    print(f"   -> Columnas finales: {list(df.columns)}")
    print(f"   -> Tipos de datos:")
    for col, dtype in df.dtypes.items():
        print(f"      • {col}: {dtype}")
    
    return df.reset_index(drop=True)


def main() -> None:
    client = crear_cliente_minio()

    print("\nDescargando datasets con los datos nuevos desde MinIO...")
    df_venta_raw    = _descargar_geoparquet(client, PATH_FUENTE, OBJ_VENTA)
    df_alquiler_raw = _descargar_geoparquet(client, PATH_FUENTE, OBJ_ALQUILER)

    print(f"\n Venta descargada:    {df_venta_raw.shape}")
    print(f" Alquiler descargada: {df_alquiler_raw.shape}")

    # PROCESAMIENTO
    print("\nProcesando datasets (limpieza + selección de variables)...")
    df_venta_proc    = preparar_dataset_completo(df_venta_raw, "venta")
    df_alquiler_proc = preparar_dataset_completo(df_alquiler_raw, "alquiler")

    # SUBIDA
    print("\nSubiendo datasets preparados a MinIO...")
    
    datasets = {
        "df_venta_xgboost": (df_venta_proc, "ventas"),
        "df_alquiler_xgboost": (df_alquiler_proc, "alquiler")
    }

    for nombre, (df, mercado) in datasets.items():
        ruta_final = f"{PATH_DESTINO_BASE}/{mercado}"
        
        subir_minio(
            df=df,
            client=client,
            path=ruta_final,
            minio_object=f"{nombre}.parquet",
        )
        print(f"{nombre}.parquet subido a {ruta_final}/")
        print(f" Dimensiones: {df.shape} | Columnas: {df.shape[1]}")

if __name__ == "__main__":
    main()