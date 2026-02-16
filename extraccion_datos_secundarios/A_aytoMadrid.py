import pandas as pd
from minio import Minio
import io
from dotenv import load_dotenv
import os


URL = "https://datos.madrid.es/egob/catalogo/300614-0-centros-educativos.csv"
BUCKET = "pd1"
OBJECT_NAME = "grupo2/raw/pruebaAtoMad.parquet"

def preparar_parquet(url: str) -> io.BytesIO:
    # Leer CSV
    df = pd.read_csv(url, sep=';', encoding= 'latin-1')

    # Convertir a parquet en memoria
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False)
    buffer.seek(0)
    return buffer

def crear_cliente_minio() -> Minio:
    # Cliente MinIO
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

def subir_minio(client: Minio, buffer: io.BytesIO):
    load_dotenv()
    minio_bucket=os.getenv("MINIO_BUCKET")
    assert minio_bucket, "Falta MINIO_BUCKET en el entorno/.env"
    assert client.bucket_exists(BUCKET), f"El bucket {BUCKET} no existe o no tienes permisos."
    # Subir objecto
    client.put_object(
        bucket_name=minio_bucket,
        object_name=OBJECT_NAME,
        data=buffer,
        length=buffer.getbuffer().nbytes
    )

if __name__ == "__main__":
    client = crear_cliente_minio()
    buffer = preparar_parquet(URL)
    subir_minio(client, buffer)