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
    "PARQUES",
    "BIBLIOTECAS",
    "PARQUES_BOMBEROS",
    "CEMENTERIOS",
    "CENTROS_DIA",
    "COMISARIAS",
    "POLIDEPORTIVOS",
    "PUNTOS_LIMPIOS",
    "IGLESIAS_CATOLICAS",
    "CENTROS_SERVICIOS_SOCIALES",
    "CENTROS_MUNICIPALES_MAYORES",
    "PISCINAS_MUNICIPALES"
]

URLS = {
"URL_EDUCACION" : "https://datos.madrid.es/dataset/300614-0-centros-educativos/resource/300614-1-centros-educativos-csv/download/300614-1-centros-educativos-csv.csv",
"URL_UNIVERSIDAD" : "https://datos.madrid.es/dataset/203166-0-universidades-educacion/resource/203166-0-universidades-educacion-csv/download/203166-0-universidades-educacion-csv.csv",
"URL_SANIDAD" : "https://datos.madrid.es/dataset/212769-0-atencion-medica/resource/212769-0-atencion-medica-csv/download/212769-0-atencion-medica-csv.csv",
"URL_LOCALES" : "https://datos.madrid.es/dataset/209548-0-censo-locales-historico/resource/209548-722-censo-locales-historico-csv/download/209548-722-censo-locales-historico-csv.csv",
"URL_PARQUES" : "https://datos.madrid.es/dataset/200761-0-parques-jardines/resource/200761-0-parques-jardines-csv/download/200761-0-parques-jardines-csv.csv",
"URL_BIBLIOTECAS" : "https://datos.madrid.es/egob/catalogo/201747-0-bibliobuses-bibliotecas.csv",
"URL_PARQUES_BOMBEROS" : "https://datos.madrid.es/egob/catalogo/211642-0-bomberos-parques.csv",
"URL_CEMENTERIOS" : "https://datos.madrid.es/egob/catalogo/205026-0-cementerios.csv",
"URL_CENTROS_DIA": "https://datos.madrid.es/egob/catalogo/200342-0-centros-dia.csv",
"URL_COMISARIAS": "https://datos.madrid.es/egob/catalogo/300600-0-comisaria.csv",
"URL_POLIDEPORTIVOS": "https://datos.madrid.es/egob/catalogo/200186-0-polideportivos.csv",
"URL_PUNTOS_LIMPIOS": "https://datos.madrid.es/egob/catalogo/200284-0-puntos-limpios-fijos.csv",
"URL_IGLESIAS_CATOLICAS": "https://datos.madrid.es/egob/catalogo/209426-0-templos-catolicas.csv",
"URL_CENTROS_SERVICIOS_SOCIALES": "https://datos.madrid.es/egob/catalogo/209094-0-centros-servicios-sociales.csv",
"URL_CENTROS_MUNICIPALES_MAYORES" : "https://datos.madrid.es/egob/catalogo/200337-0-centros-mayores.csv",
"URL_PISCINAS_MUNICIPALES" : "https://datos.madrid.es/egob/catalogo/210227-0-piscinas-publicas.csv",
}

OBJECTS = {
"OBJECT_EDUCACION" : "grupo2/raw/centros_educativos.parquet",
"OBJECT_UNIVERSIDAD" : "grupo2/raw/universidades.parquet",
"OBJECT_SANIDAD" : "grupo2/raw/hospitales.parquet",
"OBJECT_LOCALES" : "grupo2/raw/locales.parquet",
"OBJECT_PARQUES" : "grupo2/raw/parques.parquet",
"OBJECT_BIBLIOTECAS" : "grupo2/raw/bibliotecas.parquet",
"OBJECT_PARQUES_BOMBEROS" : "grupo2/raw/parques_bomberos.parquet",
"OBJECT_CEMENTERIOS" : "grupo2/raw/cementerios.parquet",
"OBJECT_CENTROS_DIA": "grupo2/raw/centros_dia.parquet",
"OBJECT_COMISARIAS": "grupo2/raw/comisarias.parquet",
"OBJECT_POLIDEPORTIVOS": "grupo2/raw/polideportivos.parquet",
"OBJECT_PUNTOS_LIMPIOS": "grupo2/raw/puntos_limpios.parquet",
"OBJECT_IGLESIAS_CATOLICAS": "grupo2/raw/iglesias_catolicas.parquet",
"OBJECT_CENTROS_SERVICIOS_SOCIALES": "grupo2/raw/centros_servicios_sociales.parquet",
"OBJECT_CENTROS_MUNICIPALES_MAYORES": "grupo2/raw/centros_municipales_mayores.parquet",
"OBJECT_PISCINAS_MUNICIPALES": "grupo2/raw/piscinas_municipales.parquet"
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