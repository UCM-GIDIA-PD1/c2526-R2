import pandas as pd
from utils.funciones_minio import crear_cliente_minio, bajar_minio, subir_minio
from utils.config import (
    DATASETS_AYTO_LIMPIEZA,
    MINIO_CLEANED_SECUNDARIOS,
    MINIO_GROUPED_SECUNDARIOS,
)

def agrupar_datasets_onehot(client, datasets_dict: dict, path_origen: str, path_destino: str, nombre_salida: str = "aytoMadrid.parquet", ) -> pd.DataFrame:
    """Agrupa los datasets del ayto de Madrid en uno y guarda el tipo en columnas onehot.

    Args:
        client (_type_): Cliente MinIO
        datasets_dict (dict): Diccionario con los nombres de los datasets y de los objetos en MinIO
        path_origen (str): Ruta dentro de MinIO de los datasets a agrupar
        path_destino (str): Ruta donde se guardará el dataset final
        nombre_salida (str, optional): Nombre del objeto en MinIO. Defaults to "aytoMadrid.parquet".

    Returns:
        pd.DataFrame: DataFrame agrupado. Columnas onehot con el siguiente esquema:
        - DATASET_UNIVERSIDADES (int8)

    """
    dfs = []

    for nombre, obj in datasets_dict.items():
        df = bajar_minio(
            client=client,
            path=path_origen,
            minio_object=obj
        )
        df["dataset"] = nombre
        dfs.append(df)
        print(f"Cargado: {nombre}")

    # 1) concatenación
    df_final = pd.concat(dfs, ignore_index=True)

    # 2) one-hot global
    df_final = pd.get_dummies(
        df_final,
        columns=["dataset"],
        prefix="DATASET",
        dtype="int8"
    )

    # 3) subir
    subir_minio(
        df=df_final,
        client=client,
        path=path_destino,
        minio_object=nombre_salida
    )
    print(f"OK: {nombre_salida}")

    return df_final


if __name__ == "__main__":
    cliente = crear_cliente_minio()

    df_resultado = agrupar_datasets_onehot(
        client=cliente,
        datasets_dict=DATASETS_AYTO_LIMPIEZA,
        path_origen=MINIO_CLEANED_SECUNDARIOS,
        path_destino=MINIO_GROUPED_SECUNDARIOS,
        nombre_salida="aytoMadrid.parquet",
    )