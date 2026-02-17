import pandas as pd
from minio import Minio
import io
from dotenv import load_dotenv
import os

tipos = [
    "EDUCACION",
    "UNIVERSIDAD",
    "SANIDAD",
    "LOCALES",
    "PARQUES"
]

URLS = {
"URL_EDUCACION" : "https://datos.madrid.es/dataset/300614-0-centros-educativos/resource/300614-1-centros-educativos-csv/download/300614-1-centros-educativos-csv.csv",
"URL_UNIVERSIDAD" : "https://datos.madrid.es/dataset/203166-0-universidades-educacion/resource/203166-0-universidades-educacion-csv/download/203166-0-universidades-educacion-csv.csv",
"URL_SANIDAD" : "https://datos.madrid.es/dataset/212769-0-atencion-medica/resource/212769-0-atencion-medica-csv/download/212769-0-atencion-medica-csv.csv",
"URL_LOCALES" : "https://datos.madrid.es/dataset/209548-0-censo-locales-historico/resource/209548-722-censo-locales-historico-csv/download/209548-722-censo-locales-historico-csv.csv",
"URL_PARQUES" : "https://datos.madrid.es/dataset/200761-0-parques-jardines/resource/200761-0-parques-jardines-csv/download/200761-0-parques-jardines-csv.csv"
}

OBJECTS = {
"OBJECT_EDUCACION" : "grupo2/raw/centros_educativos.parquet",
"OBJECT_UNIVERSIDAD" : "grupo2/raw/universidades.parquet",
"OBJECT_SANIDAD" : "grupo2/raw/hospitales.parquet",
"OBJECT_LOCALES" : "grupo2/raw/locales.parquet",
"OBJECT_PARQUES" : "grupo2/raw/parques.parquet"
}

def preparar_parquet(url: str) -> io.BytesIO:
    '''Lee un fichero csv y lo guarda en memoria'''
    df = pd.read_csv(url, sep=';', encoding= 'latin-1', dtype=str, low_memory=False)
    # Convertir a parquet en memoria
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False)
    buffer.seek(0)
    return buffer

def crear_cliente_minio() -> Minio:
    '''Crea cliente MinIO'''
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

def subir_minio(client: Minio, buffer: io.BytesIO, minio_object: str):
    '''Sube el contenido del buffer a MinIO'''
    load_dotenv()
    minio_bucket=os.getenv("MINIO_BUCKET")
    assert minio_bucket, "Falta MINIO_BUCKET en el entorno/.env"
    assert client.bucket_exists(minio_bucket), (
        f"El bucket {minio_bucket} no existe o no tienes permisos."
    )
    
    client.put_object(
        bucket_name=minio_bucket,
        object_name=minio_object,
        data=buffer,
        length=buffer.getbuffer().nbytes
    )

if __name__ == "__main__":
    client = crear_cliente_minio()
    for tipo in tipos:
        buffer = preparar_parquet(URLS[f'URL_{tipo}'])
        subir_minio(client, buffer, OBJECTS[f'OBJECT_{tipo}'])