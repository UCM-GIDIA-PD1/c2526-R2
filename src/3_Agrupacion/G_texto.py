from utils.funciones_minio import *
from utils.config import PATH_PRIMARIOS_LIMPIO, PATH_DATASETS_MODELOS
import pandas as pd

def preparar_dataset_texto(dfVenta: pd.DataFrame, dfAlquiler: pd.DataFrame) -> pd.DataFrame:
    """Dados los objetos de Venta y Alquiler, descarga los datos de MinIO, y prepara el dataset que se usará en los modelos de texto.

    Args:
        objetoVenta (str): objeto datos Venta
        objetoAlquiler (str): objeto datos Alquiler

    Returns:
        pd.DataFrame: Dataframe para los modelos de texto
    """
    dfVenta["venta"] = 1
    dfVenta["alquiler"] = 0

    dfAlquiler["venta"] = 0
    dfAlquiler["alquiler"] = 1

    df = pd.concat([dfVenta, dfAlquiler], ignore_index=True)
    df = df[["id", "Descripcion", "Anuncia", "venta", "alquiler"]]
    df["Anuncia"] = df["Anuncia"].replace({
        "Agente Pro": "Intermediario",
        "Profesional": "Intermediario"
    })
    df = pd.get_dummies(df, columns=["Anuncia"])

    return df

def main() -> None:
    cliente = crear_cliente_minio()
    OBJ_VIVIENDAS_VENTA = "viviendas_venta.parquet"
    OBJ_VIVIENDAS_ALQUILER = "viviendas_alquiler.parquet"
    OBJ_TEXTO = "texto.parquet"
    df_venta = bajar_minio(client=cliente, path=PATH_PRIMARIOS_LIMPIO, minio_object=OBJ_VIVIENDAS_VENTA)
    print(f"BAJADO: {OBJ_VIVIENDAS_VENTA}")
    df_alquiler = bajar_minio(client=cliente, path=PATH_PRIMARIOS_LIMPIO, minio_object=OBJ_VIVIENDAS_ALQUILER)
    print(f"BAJADO: {OBJ_VIVIENDAS_ALQUILER}")
    df_texto = preparar_dataset_texto(dfVenta=df_venta, dfAlquiler=df_alquiler)
    subir_minio(df=df_texto, client=cliente, path=PATH_DATASETS_MODELOS, minio_object=OBJ_TEXTO)
    print(f"OK: {OBJ_TEXTO}")


if __name__ == "__main__":
    main()