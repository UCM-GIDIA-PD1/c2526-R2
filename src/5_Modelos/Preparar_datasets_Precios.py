from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer


#Configuración de imports del proyecto
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.funciones_minio import crear_cliente_minio, subir_minio, bajar_minio



#Parámetros del script
OBJ_VIVIENDAS_VENTA = "viviendas_venta"
OBJ_VIVIENDAS_ALQUILER = "viviendas_alquiler"

PATH_REJILLAS = "rejillas"
PATH_DATASET_PRECIOS = "dataset_precios"

TARGET = "Precio"

COLS_FUGA = [
    "id",
    "Nombre",
    "Direccion",
    "Tipo_OSM",
    "geometry",
    "Precio_m2",
    "Media_precio_venta",
    "Media_precio_m2_venta",
    "Media_precio_alquiler",
    "Media_precio_m2_alquiler",
]

COLS_ALTA_CARD = ["Calle", "Tipo_Via", "Barrio", "dist_al_edificio"]

COLS_MULTICOL_NOTEBOOK = [
    "dist_min_piscinas",
    "cantidad_piscinas_cerca",
    "estaciones_cerca",
    "cantidad_alimentacion_cerca",
]


#Utilidades generales
def cargar_dataset(client, nombre_objeto: str) -> pd.DataFrame:
    candidatos = [nombre_objeto, f"{nombre_objeto}.parquet"]

    ultimo_error = None
    for candidato in candidatos:
        try:
            df = bajar_minio(client, PATH_REJILLAS, candidato)
            return pd.DataFrame(df).copy()
        except Exception as e:
            ultimo_error = e

    raise FileNotFoundError(
        f"No se pudo cargar el objeto '{nombre_objeto}' en MinIO "
        f"bajo la ruta '{PATH_REJILLAS}'. Error: {ultimo_error}"
    )


def drop_existing(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    cols_presentes = [c for c in cols if c in df.columns]
    return df.drop(columns=cols_presentes, errors="ignore")


def eliminar_columnas_por_nulos_y_varianza(df: pd.DataFrame) -> pd.DataFrame:
    cols_nulos = df.columns[df.isnull().mean() > 0.40].tolist()

    cols_zero_var = []
    for col in df.columns:
        serie = df[col].dropna()
        if serie.empty:
            cols_zero_var.append(col)
            continue
        if serie.value_counts(normalize=True).iloc[0] > 0.99:
            cols_zero_var.append(col)

    cols_drop = list(dict.fromkeys(cols_nulos + cols_zero_var))
    return df.drop(columns=cols_drop, errors="ignore")


def limpiar_base(df: pd.DataFrame, mercado: str) -> pd.DataFrame:
    df_limpio = df.copy()

    if TARGET not in df_limpio.columns:
        raise ValueError(f"El dataset no contiene la variable objetivo '{TARGET}'.")

    criticas = [c for c in [TARGET, "lat", "lon", "Superficie"] if c in df_limpio.columns]
    if criticas:
        df_limpio = df_limpio.dropna(subset=criticas)

    subset_dup = [c for c in ["lat", "lon", TARGET, "Superficie"] if c in df_limpio.columns]
    if subset_dup:
        df_limpio = df_limpio.drop_duplicates(subset=subset_dup, keep="first")

    q01 = df_limpio[TARGET].quantile(0.01)
    q99 = df_limpio[TARGET].quantile(0.99)
    df_limpio = df_limpio[df_limpio[TARGET].between(q01, q99)].copy()

    df_limpio = drop_existing(df_limpio, COLS_FUGA)
    df_limpio = eliminar_columnas_por_nulos_y_varianza(df_limpio)
    df_limpio = drop_existing(df_limpio, COLS_ALTA_CARD)

    if mercado.lower() == "alquiler" and "Cocina" in df_limpio.columns:
        df_limpio = df_limpio.drop(columns=["Cocina"], errors="ignore")

    df_limpio = drop_existing(df_limpio, COLS_MULTICOL_NOTEBOOK)
    df_limpio = df_limpio.dropna(subset=[TARGET]).reset_index(drop=True)

    if df_limpio.empty:
        raise ValueError(f"El dataset base de '{mercado}' quedó vacío tras la limpieza.")

    return df_limpio


def filtrar_individuos_knn(df: pd.DataFrame) -> pd.DataFrame:
    df_knn = df.copy()
    columnas_clave = [
        c for c in [TARGET, "Superficie", "Num_habitaciones", "Banyos", "lat", "lon"]
        if c in df_knn.columns
    ]

    for col in columnas_clave:
        if pd.api.types.is_numeric_dtype(df_knn[col]):
            q01, q99 = df_knn[col].quantile([0.01, 0.99])
            df_knn = df_knn[df_knn[col].between(q01, q99)]

    df_knn = df_knn.reset_index(drop=True)

    if df_knn.empty:
        raise ValueError("El dataset de kNN quedó vacío tras filtrar individuos extremos.")

    return df_knn


def _separar_tipos(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    features = [c for c in df.columns if c != TARGET]
    num_cols = df[features].select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = df[features].select_dtypes(include=["bool"]).columns.tolist()
    cat_cols = [c for c in features if c not in num_cols + bool_cols]
    return num_cols, bool_cols, cat_cols


def _filtrar_categoricas_baja_cardinalidad(df: pd.DataFrame, max_unique: int) -> List[str]:
    _, bool_cols, cat_cols = _separar_tipos(df)

    cols_validas = []
    for col in bool_cols + cat_cols:
        nunique = df[col].nunique(dropna=True)
        if nunique <= max_unique:
            cols_validas.append(col)

    return cols_validas


def _one_hot_dataframe(df: pd.DataFrame, cat_cols: List[str], dummy_na: bool = False) -> pd.DataFrame:
    base = df.copy()
    cat_cols_existentes = [c for c in cat_cols if c in base.columns]

    for col in cat_cols_existentes:
        base[col] = base[col].astype("string").fillna("DESCONOCIDO")

    if cat_cols_existentes:
        base = pd.get_dummies(
            base,
            columns=cat_cols_existentes,
            drop_first=True,
            dummy_na=dummy_na,
        )

    return base


def _imputar_numericas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    imp = SimpleImputer(strategy="median")
    out.loc[:, :] = imp.fit_transform(out)
    return out


def _quitar_multicolinealidad_por_objetivo(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.85,
) -> pd.DataFrame:
    if X.shape[1] <= 1:
        return X

    corr_target = X.apply(lambda s: s.corr(y), axis=0).abs().fillna(0.0)
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    drop_cols = set()
    for col in upper.columns:
        relacionadas = upper.index[upper[col] > threshold].tolist()
        for otra in relacionadas:
            if col in drop_cols or otra in drop_cols:
                continue

            if corr_target.get(col, 0) >= corr_target.get(otra, 0):
                drop_cols.add(otra)
            else:
                drop_cols.add(col)

    X_final = X.drop(columns=list(drop_cols), errors="ignore")
    return X_final if not X_final.empty else X



#Preparación por modelo
def preparar_dataset_regresion(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    y = work[TARGET].copy()

    num_cols, _, _ = _separar_tipos(work)
    cat_cols = _filtrar_categoricas_baja_cardinalidad(work, max_unique=20)

    cols_modelo = sorted(set(num_cols + cat_cols + [TARGET]))
    work = work[cols_modelo].copy()

    X = work.drop(columns=[TARGET])
    X = _one_hot_dataframe(X, [c for c in cat_cols if c in X.columns])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = _imputar_numericas(X)

    if X.shape[1] == 0:
        raise ValueError("No hay variables válidas para construir el dataset de regresión.")

    f_vals, p_vals = f_regression(X, y)
    ranking = pd.DataFrame(
        {"feature": X.columns, "f_value": f_vals, "p_value": p_vals}
    ).sort_values(["p_value", "f_value"], ascending=[True, False])

    seleccionadas = ranking.loc[ranking["p_value"] < 0.05, "feature"].tolist()
    if len(seleccionadas) < 8:
        seleccionadas = ranking["feature"].head(min(25, len(ranking))).tolist()

    X = X[seleccionadas].copy()
    X = _quitar_multicolinealidad_por_objetivo(X, y, threshold=0.85)

    return pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)


def preparar_dataset_knn(df: pd.DataFrame) -> pd.DataFrame:
    work = filtrar_individuos_knn(df)
    y = work[TARGET].copy()

    num_cols, bool_cols, _ = _separar_tipos(work)
    cols_modelo = sorted(set(num_cols + bool_cols + [TARGET]))
    work = work[cols_modelo].copy()

    X = work.drop(columns=[TARGET]).copy()

    for col in bool_cols:
        if col in X.columns:
            X[col] = X[col].astype(int)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = _imputar_numericas(X)

    if X.shape[1] == 0:
        raise ValueError("No hay variables válidas para construir el dataset de kNN.")

    mi = mutual_info_regression(X, y, random_state=42)
    ranking = pd.DataFrame({"feature": X.columns, "mi": mi}).sort_values("mi", ascending=False)

    umbral = max(0.001, float(ranking["mi"].median()))
    seleccionadas = ranking.loc[ranking["mi"] >= umbral, "feature"].tolist()

    if len(seleccionadas) < 6:
        seleccionadas = ranking["feature"].head(min(15, len(ranking))).tolist()

    X = X[seleccionadas].copy()
    X = _quitar_multicolinealidad_por_objetivo(X, y.loc[X.index], threshold=0.90)

    return pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)


def preparar_dataset_arboles(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    y = work[TARGET].copy()

    num_cols, _, _ = _separar_tipos(work)
    cat_cols = _filtrar_categoricas_baja_cardinalidad(work, max_unique=35)

    cols_modelo = sorted(set(num_cols + cat_cols + [TARGET]))
    work = work[cols_modelo].copy()

    X = work.drop(columns=[TARGET])
    X = _one_hot_dataframe(X, [c for c in cat_cols if c in X.columns])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = _imputar_numericas(X)

    if X.shape[1] == 0:
        raise ValueError("No hay variables válidas para construir el dataset de árboles.")

    mi = mutual_info_regression(X, y, random_state=42)
    ranking = pd.DataFrame({"feature": X.columns, "mi": mi}).sort_values("mi", ascending=False)

    seleccionadas = ranking.loc[ranking["mi"] > 0, "feature"].tolist()
    if len(seleccionadas) < 10:
        seleccionadas = ranking["feature"].head(min(40, len(ranking))).tolist()
    else:
        seleccionadas = ranking["feature"].head(min(max(20, len(seleccionadas)), len(ranking))).tolist()

    X = X[seleccionadas].copy()

    return pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)


#Construcción final
def construir_datasets_modelos(client) -> Dict[str, pd.DataFrame]:
    df_ventas_base = limpiar_base(cargar_dataset(client, OBJ_VIVIENDAS_VENTA), "venta")
    df_alquiler_base = limpiar_base(cargar_dataset(client, OBJ_VIVIENDAS_ALQUILER), "alquiler")

    return {
        "df_ventas_regresion": preparar_dataset_regresion(df_ventas_base),
        "df_alquiler_regresion": preparar_dataset_regresion(df_alquiler_base),
        "df_ventas_knn": preparar_dataset_knn(df_ventas_base),
        "df_alquiler_knn": preparar_dataset_knn(df_alquiler_base),
        "df_ventas_arboles": preparar_dataset_arboles(df_ventas_base),
        "df_alquiler_arboles": preparar_dataset_arboles(df_alquiler_base),
    }



# Subida a MinIO
def subir_datasets_a_minio(datasets: Dict[str, pd.DataFrame], client) -> None:
    for nombre, df in datasets.items():
        mercado = "ventas" if nombre.startswith("df_ventas_") else "alquiler"
        filename = f"{nombre}.parquet"

        subir_minio(
            df=df,
            client=client,
            path=f"{PATH_DATASET_PRECIOS}/{mercado}",
            minio_object=filename,
        )


def mostrar_resumen(datasets: Dict[str, pd.DataFrame]) -> None:
    print("\nResumen de datasets generados")
    print("=" * 90)
    for nombre, df in datasets.items():
        print(f"{nombre:<24} -> filas={df.shape[0]:>7} | columnas={df.shape[1]:>5}")
    print("=" * 90)

    print("\nSubida a MinIO:")
    print(f"- {PATH_DATASET_PRECIOS}/ventas")
    print(f"- {PATH_DATASET_PRECIOS}/alquiler")


def main() -> None:
    client = crear_cliente_minio()
    datasets = construir_datasets_modelos(client)
    subir_datasets_a_minio(datasets, client)
    mostrar_resumen(datasets)


if __name__ == "__main__":
    main()