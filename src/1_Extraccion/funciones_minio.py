'''
funciones_minio.py

Script de funciones para interactuar con el servidor de MinIO que usaremos repetidamente 
a lo largo del proyecto.

'''

from minio import Minio
import os
from dotenv import load_dotenv
import io
import pandas as pd


def crear_cliente_minio() -> Minio:
    """ Inicializa un cliente MinIO a partir de las variables de entorno en .env
    (MINIO_ACCESS_KEY, MINIO_SECRET_KEY y MINIO_ENDPOINT).

    Returns:
        Minio: Cliente de MinIO
    """
    load_dotenv()
    minio_access_key=os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key=os.getenv("MINIO_SECRET_KEY")
    minio_endpoint=os.getenv("MINIO_ENDPOINT")

    assert minio_access_key, "Falta MINIO_ACCESS_KEY en el entorno/.env"
    assert minio_secret_key, "Falta MINIO_SECRET_KEY en el entorno/.env"
    assert minio_endpoint, "Falta MINIO_ENDPOINT en el entorno/.env"

    return Minio(
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=True
    )

def minio_subir_memoria(client: Minio, buffer: io.BytesIO, minio_object: str) -> None:
    """Sube a MinIO el contenido de un buffer en memoria como un objeto.

    Args:
        client (Minio): Cliente MinIO ya inicializado
        buffer (io.BytesIO): Buffer en memoria con los datos a subir
        minio_object (str): Nombre del objeto destino en el bucket
    """
    
    load_dotenv()
    minio_bucket=os.getenv("MINIO_BUCKET")
    assert minio_bucket, "Falta MINIO_BUCKET en el entorno/.env"
    assert client.bucket_exists(minio_bucket), (
        f"El bucket {minio_bucket} no existe o no tienes permisos."
    )
    
    buffer.seek(0)
    length = buffer.getbuffer().nbytes

    client.put_object(
        bucket_name=minio_bucket,
        object_name=minio_object,
        data=buffer,
        length=length,
        content_type="application/octet-stream"
    )

def bajar_minio(client: Minio, minio_object: str) -> pd.DataFrame:
    """ 
    Descarga desde MinIO un dataframe

    Args:
        client (Minio): Cliente de MinIO ya inicializado
        minio_object (str): Nombre del objeto de origen en el bucket

    Returns:
        pd.DataFrame: Dataframe solicitado
    """
    load_dotenv()
    minio_bucket=os.getenv("MINIO_BUCKET")
    assert minio_bucket, "Falta MINIO_BUCKET en el entorno/.env"
    assert client.bucket_exists(minio_bucket), (
        f"El bucket {minio_bucket} no existe o no tienes permisos."
    )

    response = client.get_object(
        bucket_name=minio_bucket,
        object_name=minio_object,
    )

    # Leer en memoria
    data = io.BytesIO(response.read())
    df = pd.read_parquet(data)

    response.close()
    response.release_conn()
    return df