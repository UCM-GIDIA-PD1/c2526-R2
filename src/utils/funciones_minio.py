"""
funciones_minio.py

Script de funciones para interactuar con el servidor de MinIO que usaremos repetidamente 
a lo largo del proyecto.

"""

from minio import Minio
import os
from dotenv import load_dotenv
import io
import urllib3
import pandas as pd
import geopandas as gpd



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

    cliente_http = urllib3.PoolManager(
    
        timeout=urllib3.Timeout(connect=10.0, read=600.0), 
        
        retries=urllib3.Retry(
            total=5, 
            backoff_factor=0.5, 
            status_forcelist=[500, 502, 503, 504]
        )
    )

    return Minio(
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=True,
        http_client = cliente_http
    )


def minio_subir_memoria(client: Minio, path: str, minio_object: str, buffer: io.BytesIO) -> None:
    """Sube a MinIO el contenido de un buffer en memoria como un objeto.

    Args:
        client (Minio): Cliente MinIO ya inicializado
        path: ruta dentro del bucket donde se guarda el objeto
        buffer (io.BytesIO): Buffer en memoria con los datos a subir
        minio_object (str): Nombre del objeto destino en el bucket
    """
    load_dotenv()
    minio_bucket=os.getenv("MINIO_BUCKET")
    minio_groupPath=os.getenv("MINIO_GROUP_PATH")
    assert minio_bucket, "Falta MINIO_BUCKET en el entorno/.env"
    assert minio_groupPath, "Falta MINIO_GROUP_PATH en el entorno/.env"
    assert client.bucket_exists(minio_bucket), (
        f"El bucket {minio_bucket} no existe o no tienes permisos."
    )
    
    buffer.seek(0)
    length = buffer.getbuffer().nbytes

    client.put_object(
        bucket_name=minio_bucket,
        object_name=f'{minio_groupPath}/{path}/{minio_object}',
        data=buffer,
        length=length,
        content_type="application/octet-stream"
    )

def subir_minio(df: pd.DataFrame, client:Minio, path: str, minio_object: str) -> None:
    """Sube dataframe a MinIO, convirtiéndolo a parquet en el proceso.

    Args:
        df (pd.DataFrame): DataFrame a subir
        client (Minio): Cliente de MinIO
        path (str): Ruta dentro del bucket donde se situará el objeto
        minio_object (str): Nombre del objeto destino en el bucket
    """
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False)
    buffer.seek(0)
    load_dotenv()
    minio_bucket=os.getenv("MINIO_BUCKET")
    minio_groupPath=os.getenv("MINIO_GROUP_PATH")
    assert minio_bucket, "Falta MINIO_BUCKET en el entorno/.env"
    assert minio_groupPath, "Falta MINIO_GROUP_PATH en el entorno/.env"
    assert client.bucket_exists(minio_bucket), (
        f"El bucket {minio_bucket} no existe o no tienes permisos."
    )
    
    buffer.seek(0)
    length = buffer.getbuffer().nbytes

    client.put_object(
        bucket_name=minio_bucket,
        object_name=f'{minio_groupPath}/{path}/{minio_object}',
        data=buffer,
        length=length,
        content_type="application/octet-stream"
    )

def subir_mapa_minio(client:Minio, gdf_mapa:gpd.GeoDataFrame,path:str, nombre_mapa:str):
    """
    Sube un GeoDataFrame a MinIO en formato GeoParquet.
    Mantiene intacta la columna 'geometry' y el CRS (Sistema de coordenadas).
    """
    minio_bucket=os.getenv("MINIO_BUCKET")
    minio_groupPath=os.getenv("MINIO_GROUP_PATH")
    assert minio_bucket, "Falta MINIO_BUCKET en el entorno/.env"
    assert minio_groupPath, "Falta MINIO_GROUP_PATH en el entorno/.env"
    assert client.bucket_exists(minio_bucket), (
        f"El bucket {minio_bucket} no existe o no tienes permisos."
    )
    ruta_guardado = f"{minio_groupPath}/{path}/{nombre_mapa}.parquet"
        
    buffer = io.BytesIO()
    gdf_mapa.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    client.put_object(
        bucket_name=minio_bucket,
        object_name=ruta_guardado,
        data=buffer,
        length=buffer.getbuffer().nbytes,
        content_type="application/octet-stream"
    )


def bajar_minio(client: Minio, path: str, minio_object: str) -> pd.DataFrame:
    """ 
    Descarga desde MinIO un dataframe

    Args:
        client (Minio): Cliente de MinIO ya inicializado
        path: Ruta dentro del bucket donde se encuentra el objeto
        minio_object (str): Nombre del objeto de origen en el bucket

    Returns:
        pd.DataFrame: Dataframe solicitado
    """

    load_dotenv()
    minio_bucket=os.getenv("MINIO_BUCKET")
    minio_groupPath=os.getenv("MINIO_GROUP_PATH")
    assert minio_bucket, "Falta MINIO_BUCKET en el entorno/.env"
    assert minio_groupPath, "Falta MINIO_GROUP_PATH en el entorno/.env"
    assert client.bucket_exists(minio_bucket), (
        f"El bucket {minio_bucket} no existe o no tienes permisos."
    )

    response = client.get_object(
        bucket_name=minio_bucket,
        object_name=f'{minio_groupPath}/{path}/{minio_object}'
    )

    # Leer en memoria
    data = io.BytesIO(response.read())
    df = pd.read_parquet(data)
    response.close()
    response.release_conn()
    return df


def bajar_minio_especifico(client: Minio, path: str, minio_object: str,columnas:list) -> pd.DataFrame:
    """ 
    Descarga desde MinIO un dataframe

    Args:
        client (Minio): Cliente de MinIO ya inicializado
        path: Ruta dentro del bucket donde se encuentra el objeto
        minio_object (str): Nombre del objeto de origen en el bucket

    Returns:
        pd.DataFrame: Dataframe solicitado
    """

    load_dotenv()
    minio_bucket=os.getenv("MINIO_BUCKET")
    minio_groupPath=os.getenv("MINIO_GROUP_PATH")
    assert minio_bucket, "Falta MINIO_BUCKET en el entorno/.env"
    assert minio_groupPath, "Falta MINIO_GROUP_PATH en el entorno/.env"
    assert client.bucket_exists(minio_bucket), (
        f"El bucket {minio_bucket} no existe o no tienes permisos."
    )
    response = client.get_object(
        bucket_name=minio_bucket,
        object_name=f'{minio_groupPath}/{path}/{minio_object}'
    )

    # Leer en memoria
    data = io.BytesIO(response.read())
    df = pd.read_parquet(data,columns=columnas)

    response.close()
    response.release_conn()
    return df


def buscar_todos_los_archivos(client: Minio, path: str)->list:
    """
    Devuelve una lista con las rutas completas de todos los archivos .parquet 
    que existen dentro de una carpeta específica en MinIO.
    
    Args:
        client: El cliente de MinIO inicializado.
        ruta_busqueda (str): La ruta dentro del bucket (ej: 'maiday/datos_primarios/venta/')
    """
    minio_bucket=os.getenv("MINIO_BUCKET")
    minio_groupPath=os.getenv("MINIO_GROUP_PATH")
    assert minio_bucket, "Falta MINIO_BUCKET en el entorno/.env"
    assert minio_groupPath, "Falta MINIO_GROUP_PATH en el entorno/.env"
    assert client.bucket_exists(minio_bucket), (
        f"El bucket {minio_bucket} no existe o no tienes permisos."
    )

    ruta_busqueda = f"{minio_groupPath}/{path}/"
        
    lista_archivos = []

    objetos = client.list_objects(minio_bucket, prefix=ruta_busqueda, recursive=True)
    for obj in objetos:
        nombre_archivo = obj.object_name
        if nombre_archivo.endswith('.parquet'):
            lista_archivos.append(nombre_archivo.removeprefix(ruta_busqueda))
                
    return lista_archivos


def bajar_mapa_minio(client:Minio, path:str,nombre_capa:str):
    """
    Descarga un GeoParquet desde MinIO y lo devuelve como un GeoDataFrame 
    listo para hacer cruces espaciales (sjoin) o pintarlo en pantalla.
    """
    minio_bucket=os.getenv("MINIO_BUCKET")
    minio_groupPath=os.getenv("MINIO_GROUP_PATH")
    assert minio_bucket, "Falta MINIO_BUCKET en el entorno/.env"
    assert minio_groupPath, "Falta MINIO_GROUP_PATH en el entorno/.env"
    assert client.bucket_exists(minio_bucket), (
        f"El bucket {minio_bucket} no existe o no tienes permisos."
    )
    
    ruta_archivo = f"{minio_groupPath}/{path}/{nombre_capa}.parquet"

    respuesta = client.get_object(minio_bucket, ruta_archivo)
    buffer = io.BytesIO(respuesta.read())
        
    gdf = gpd.read_parquet(buffer)

    respuesta.close()
    respuesta.release_conn() 

    return gdf
        
    