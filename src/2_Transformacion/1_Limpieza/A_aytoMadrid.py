import pandas as pd
from minio import Minio
from dotenv import load_dotenv
import os
import io

tipos = [
    "centros_educativos",
    "universidades",
    "locales",
    "hospitales",
    "parques"
]

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

def subir_minio(client: Minio, buffer: io.BytesIO, minio_object: str) -> None:
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



def bajar_minio(client: Minio, minio_object: str) -> pd.DataFrame:
    '''Obtiene un dataframe de un objeto de Minio'''
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




if __name__ == "__main__":
    client = crear_cliente_minio()
    for tipo in tipos:
        minio_object=f"grupo2/raw/{tipo}.parquet"
        df = bajar_minio(client, minio_object)
        print(df.head())