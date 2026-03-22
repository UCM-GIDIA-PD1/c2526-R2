from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
from utils.funciones_minio import crear_cliente_minio, subir_minio

import sys
import nbformat
import pandas as pd

# Configuración de imports del proyecto
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# Parámetros
TARGET = "Precio"
PATH_DATASET_PRECIOS = "dataset_ml/precios"
PATH_VENTAS = f"{PATH_DATASET_PRECIOS}/ventas"
PATH_ALQUILER = f"{PATH_DATASET_PRECIOS}/alquiler"

NOTEBOOK_PATH = ROOT / "src" / "4_Analisis" / "analisis_estadistico_2.ipynb"




def ejecutar_notebook_y_extraer_dataframes(notebook_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el notebook y extrae df_venta_limpio y df_alquiler_limpio
    de su namespace final.
    """
    if not notebook_path.exists():
        raise FileNotFoundError(f"No existe el notebook: {notebook_path}")

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    namespace = {
        "__name__": "__notebook_exec__",
        "__file__": str(notebook_path),
    }

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        source = cell.source
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

    df_venta_limpio = namespace["df_venta_limpio"].copy()
    df_alquiler_limpio = namespace["df_alquiler_limpio"].copy()

    if not isinstance(df_venta_limpio, pd.DataFrame):
        raise TypeError("'df_venta_limpio' no es un DataFrame.")
    if not isinstance(df_alquiler_limpio, pd.DataFrame):
        raise TypeError("'df_alquiler_limpio' no es un DataFrame.")

    return df_venta_limpio, df_alquiler_limpio


def construir_datasets_modelos(
    df_venta_limpio: pd.DataFrame,
    df_alquiler_limpio: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Crea los 8 datasets finales:
    - 2 base: df_venta_limpio, df_alquiler_limpio
    - 6 para modelos: 3 ventas + 3 alquiler

    """
    datasets = {
        # Bases limpias
        "df_venta_limpio": df_venta_limpio.copy(),
        "df_alquiler_limpio": df_alquiler_limpio.copy(),

        # Ventas
        "df_ventas_regresion": df_venta_limpio.copy(),
        "df_ventas_arboles": df_venta_limpio.copy(),
        "df_ventas_knn": df_venta_limpio.copy(),

        # Alquiler
        "df_alquiler_regresion": df_alquiler_limpio.copy(),
        "df_alquiler_arboles": df_alquiler_limpio.copy(),
        "df_alquiler_knn": df_alquiler_limpio.copy(),
    }

    return datasets


def subir_datasets_a_minio(datasets: Dict[str, pd.DataFrame], client) -> None:
    """
    Sube cada dataset a su ruta correcta:
    - ventas -> dataset_ml/precios/ventas
    - alquiler -> dataset_ml/precios/alquiler
    """
    for nombre, df in datasets.items():
        if nombre.startswith("df_venta"):
            path_destino = PATH_VENTAS
        elif nombre.startswith("df_ventas_"):
            path_destino = PATH_VENTAS
        elif nombre.startswith("df_alquiler"):
            path_destino = PATH_ALQUILER
        else:
            raise ValueError(f"No se pudo inferir la ruta MinIO para '{nombre}'.")

        filename = f"{nombre}.parquet"

        subir_minio(
            df=df,
            client=client,
            path=path_destino,
            minio_object=filename,
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