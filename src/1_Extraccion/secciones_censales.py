import os
import requests
import geopandas as gdp
import io
from dotenv import load_dotenv
from funciones_minio import crear_cliente_minio, minio_subir_memoria

'''
Script para la extracción de la cartografía de secciones censales del Ayuntamiento de Madrid.
Permite obtener las geometrías necesarias para representar datos estadísticos por zonas.
'''

def descargar_secciones_censales():
    """ 
    Descarga las secciones censales del Geoportal del Ayuntamiento de Madrid en formato TopoJSON,
    genera el código CUSEC único para cada sección y sube el resultado en formato Parquet a MinIO.

    Raises:
        requests.exceptions.RequestException: Si la conexión con el Geoportal del Ayuntamiento de Madrid falla.
    """
    print("Iniciando proceso de extracción de las secciones censales del Ayuntamiento de Madrid... \n")
    url = "https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Seccionado/TopoJSON/Secciones_Censales.json"
    
    print("Descargando datos de las secciones censales... \n")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"ERROR: {response.status_code}")
        return None
    
    print("Procesando datos de las secciones censales... \n")
    gdf = gdp.read_file(io.BytesIO(response.content))
    
    # Nos quedamos solo con el código de sección y la geometría
    # Se utiliza COD_SECCIO para construir el CUSEC (Madrid 28 + Municipio 079 + Código Sección)
    gdf['CUSEC'] = "28079" + gdf['COD_SECCIO'].astype(str).str.zfill(5)
    gdf_final = gdf[['geometry', 'CUSEC']].copy()

    # Guardamos en Parquet
    buffer = io.BytesIO()
    gdf_final.to_parquet(buffer, index=False)
    buffer.seek(0)

    cliente = crear_cliente_minio()
    minio_subir_memoria(cliente, "datos_secundarios/secciones_censales", "secciones_censales_madrid.parquet", buffer)
    print("Archivo de secciones censales subido a MinIO \n")

if __name__ == "__main__":
    descargar_secciones_censales()