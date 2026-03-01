import requests
import geopandas as gdp
import io
from minio import Minio
from src.utils.funciones_minio import crear_cliente_minio, minio_subir_memoria
from src.config import URL_SECCIONES,URL_BARRIOS, MINIO_REJILLAS_SUCIO, OBJ_SECCIONES, OBJ_BARRIOS

'''
Script para la extracción de la cartografía de secciones censales y barrios del Ayuntamiento de Madrid.
Permite obtener las geometrías necesarias para representar datos estadísticos por zonas.
'''

def descargar_barrios(cliente:Minio):
    print("Iniciando proceso de extracción del mapa de barrios del Ayuntamiento de Madrid... \n")
    
    print("Descargando datos de los barrios... \n")
    response = requests.get(URL_BARRIOS)
    
    if response.status_code != 200:
        print(f"ERROR: {response.status_code}")
        return None
    
    gdf_barrios = gdp.read_file(f"zip+https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Barrios.zip")
    gdf_final = gdf_barrios[["COD_BAR","NOMBRE","AREA","geometry"]]

    # Guardamos en Parquet
    buffer = io.BytesIO()
    gdf_final.to_parquet(buffer, index=False)
    buffer.seek(0)

    minio_subir_memoria(cliente, MINIO_REJILLAS_SUCIO, OBJ_BARRIOS, buffer)
    print("Archivo de secciones censales subido a MinIO \n")



def descargar_secciones_censales(cliente:Minio):
    """ 
    Descarga las secciones censales del Geoportal del Ayuntamiento de Madrid en formato TopoJSON,
    genera el código CUSEC único para cada sección y sube el resultado en formato Parquet a MinIO.

    Raises:
        requests.exceptions.RequestException: Si la conexión con el Geoportal del Ayuntamiento de Madrid falla.
    """
    print("Iniciando proceso de extracción de las secciones censales del Ayuntamiento de Madrid... \n")
    
    print("Descargando datos de las secciones censales... \n")
    response = requests.get(URL_SECCIONES)
    
    if response.status_code != 200:
        print(f"ERROR: {response.status_code}")
        return None
    
    print("Procesando datos de las secciones censales... \n")
    gdf = gdp.read_file(io.BytesIO(response.content))

    # Nos quedamos solo con el código de sección y la geometría
    # Se utiliza COD_SECCIO para construir el CUSEC (Madrid 28 + Municipio 079 + Código Sección)
    gdf['CUSEC'] = "28079" + gdf['COD_SECCIO'].astype(str).str.zfill(5)
    gdf_final = gdf[['geometry', 'CUSEC','COD_BAR','Area']].copy()
    gdf_final["AREA"] = gdf_final["Area"]
    gdf_final = gdf_final.drop(columns=['Area'])

    # Guardamos en Parquet
    buffer = io.BytesIO()
    gdf_final.to_parquet(buffer, index=False)
    buffer.seek(0)

    minio_subir_memoria(cliente, MINIO_REJILLAS_SUCIO, OBJ_SECCIONES, buffer)
    print("Archivo de secciones censales subido a MinIO \n")

def descargar_mapas():
    cliente = crear_cliente_minio()
    descargar_barrios(cliente)
    descargar_secciones_censales(cliente)


if __name__ == "__main__":
    descargar_mapas()
