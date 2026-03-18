from utils.funciones_minio import *
from utils.config import PATH_PRIMARIOS_LIMPIO
import pandas as pd

def preparar_dataset_texto(objetoVenta: str, objetoAlquiler: str) -> pd.DataFrame:
    """Dados los objetos de Venta y Alquiler, descarga los datos de MinIO, y prepara el dataset que se usará en los modelos de texto.

    Args:
        objetoVenta (str): objeto datos Venta
        objetoAlquiler (str): objeto datos Alquiler

    Returns:
        pd.DataFrame: Dataframe para los modelos de texto
    """
    client = crear_cliente_minio()
    df_venta = bajar_minio(client, PATH_PRIMARIOS_LIMPIO, objetoVenta)
    df_alquiler = bajar_minio(client, PATH_PRIMARIOS_LIMPIO, objetoAlquiler)
    df_venta["venta"] = 1
    df_venta["alquiler"] = 0

    df_alquiler["venta"] = 0
    df_alquiler["alquiler"] = 1

    df = pd.concat([df_venta, df_alquiler], ignore_index=True)
    df = df[["id", "Descripcion", "Anuncia", "venta", "alquiler"]]
    df["Anuncia"] = df["Anuncia"].replace({
        "Agente Pro": "Intermediario",
        "Profesional": "Intermediario"
    })
    df = pd.get_dummies(df, columns=["Anuncia"])

    return df