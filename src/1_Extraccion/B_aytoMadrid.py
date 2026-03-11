import pandas as pd
import io
from utils.funciones_minio import crear_cliente_minio, minio_subir_memoria
from utils.config import DATASETS_AYTO, MINIO_RAW_SECUNDARIOS


def preparar_parquet(url: str) -> io.BytesIO:
    """
    Lee un csv, lo guarda en memoria y lo convierte a parquet(en memoria)

    Args:
        url (str): url que descarga el csv

    Returns:
        io.BytesIO: buffer de memoria
    """
    df = pd.read_csv(url, sep=';', encoding= 'latin-1', dtype=str, low_memory=False)
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False)
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    for nombre, cfg in DATASETS_AYTO.items():
        buffer = preparar_parquet(cfg["url"])
        minio_subir_memoria(client=cliente, path=MINIO_RAW_SECUNDARIOS, buffer=buffer, minio_object=cfg["object"])
        print(f"OK: {nombre}")
