from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")

import pandas as pd


# Configuración de imports del proyecto
ROOT = Path(__file__).resolve().parents[2]
print(f"ROOT detectado: {ROOT}")

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    print(f"Añadido al path: {ROOT}")

from utils.funciones_minio import crear_cliente_minio, subir_minio


# Parámetros
PATH_DATASET_PRECIOS = "dataset_ml/precios"
PATH_VENTAS = f"{PATH_DATASET_PRECIOS}/ventas"
PATH_ALQUILER = f"{PATH_DATASET_PRECIOS}/alquiler"

NOTEBOOK_PATH = ROOT / "src" / "4_Analisis" / "analisis_estadistico_2.ipynb"
print(f"Notebook path: {NOTEBOOK_PATH}")


def ejecutar_notebook_y_extraer_dataframes(notebook_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lee el notebook .ipynb como JSON, ejecuta sus celdas de código
    y extrae df_venta_limpio y df_alquiler_limpio.
    """
    if not notebook_path.exists():
        raise FileNotFoundError(f"No existe el notebook: {notebook_path}")

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    try:
        from IPython.display import display
    except Exception:
        def display(*args, **kwargs):
            return None

    namespace = {
        "__name__": "__notebook_exec__",
        "__file__": str(notebook_path),
        "display": display,
    }

    for i, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        if isinstance(source, list):
            source = "".join(source)

        if not str(source).strip():
            continue

        # Saltar magias de Jupyter si las hubiera
        lineas = []
        for linea in source.splitlines():
            stripped = linea.strip()
            if stripped.startswith("%") or stripped.startswith("!"):
                continue
            lineas.append(linea)
        source = "\n".join(lineas)

        if not source.strip():
            continue

        try:
            exec(compile(source, filename=f"{notebook_path.name}::cell_{i}", mode="exec"), namespace)
        except Exception as e:
            raise RuntimeError(
                f"Error ejecutando la celda {i} del notebook '{notebook_path.name}': {e}"
            ) from e

    if "df_venta_limpio" not in namespace:
        raise KeyError("No se encontró 'df_venta_limpio' en el notebook.")
    if "df_alquiler_limpio" not in namespace:
        raise KeyError("No se encontró 'df_alquiler_limpio' en el notebook.")

    df_venta_limpio = namespace["df_venta_limpio"]
    df_alquiler_limpio = namespace["df_alquiler_limpio"]

    if not isinstance(df_venta_limpio, pd.DataFrame):
        raise TypeError("'df_venta_limpio' no es un DataFrame.")
    if not isinstance(df_alquiler_limpio, pd.DataFrame):
        raise TypeError("'df_alquiler_limpio' no es un DataFrame.")

    return df_venta_limpio.copy(), df_alquiler_limpio.copy()


def construir_datasets_modelos(
    df_venta_limpio: pd.DataFrame,
    df_alquiler_limpio: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Crea 8 datasets:
    - 2 base
    - 6 datasets de modelos
    """
    return {
        "df_venta_limpio": df_venta_limpio.copy(),
        "df_alquiler_limpio": df_alquiler_limpio.copy(),

        "df_ventas_regresion": df_venta_limpio.copy(),
        "df_ventas_arboles": df_venta_limpio.copy(),
        "df_ventas_knn": df_venta_limpio.copy(),

        "df_alquiler_regresion": df_alquiler_limpio.copy(),
        "df_alquiler_arboles": df_alquiler_limpio.copy(),
        "df_alquiler_knn": df_alquiler_limpio.copy(),
    }


def subir_datasets_a_minio(datasets: Dict[str, pd.DataFrame], client) -> None:
    """
    Sube cada dataset a su ruta correcta:
    - ventas -> dataset_ml/precios/ventas
    - alquiler -> dataset_ml/precios/alquiler
    """
    for nombre, df in datasets.items():
        if nombre.startswith("df_venta"):
            path_destino = PATH_VENTAS
        elif nombre.startswith("df_alquiler"):
            path_destino = PATH_ALQUILER
        else:
            raise ValueError(f"No se pudo inferir la ruta MinIO para '{nombre}'.")

        subir_minio(
            df=df,
            client=client,
            path=path_destino,
            minio_object=f"{nombre}.parquet",
        )


def mostrar_resumen(datasets: Dict[str, pd.DataFrame]) -> None:
    print("\nResumen de datasets generados")
    print("=" * 95)
    for nombre, df in datasets.items():
        print(f"{nombre:<24} -> filas={df.shape[0]:>7} | columnas={df.shape[1]:>5}")
    print("=" * 95)

    print("\nSubida a MinIO:")
    print(f"- {PATH_VENTAS}")
    print(f"- {PATH_ALQUILER}")


def main() -> None:
    client = crear_cliente_minio()

    df_venta_limpio, df_alquiler_limpio = ejecutar_notebook_y_extraer_dataframes(NOTEBOOK_PATH)

    datasets = construir_datasets_modelos(
        df_venta_limpio=df_venta_limpio,
        df_alquiler_limpio=df_alquiler_limpio,
    )

    subir_datasets_a_minio(datasets, client)
    mostrar_resumen(datasets)


if __name__ == "__main__":
    main()