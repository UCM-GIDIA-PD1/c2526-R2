import pandas as pd
import requests
import zipfile
import geopandas as gpd
import io
from utils.funciones_minio import crear_cliente_minio, minio_subir_memoria
from utils.config import URL_CATASTRO, MINIO_CATASTRO, OBJ_CATASTRO

'''
Script para la extracción de los datos del Catastro de Madrid, incluyendo geometrías de edificios y su año de construcción.
'''

def descargar_catastro():
    """
    Extrae los edificios del Catastro, con su geometría, procesa el año de construcción 
    y sube los datos espaciales a MinIO en formato Parquet.

    Raises:
        requests.exceptions.RequestException: Si la conexión con el catastro falla.
    """
    print("Iniciando proceso de extracción del catastro... \n")
    
    print("Descargando datos del catastro... \n")
    resp = requests.get(URL_CATASTRO)
    if resp.status_code != 200: 
        print(f"ERROR: {resp.status_code}")
        return
    
    print("Procesando datos del catastro... \n")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        archivo_gml = [f for f in z.namelist() if f.endswith('building.gml')][0]
        with z.open(archivo_gml) as f:
            gdf = gpd.read_file(f)

    print("Sacando año de construcción... \n")
    columnas_utiles = ['geometry', 'beginning']
    gdf = gdf[[c for c in columnas_utiles if c in gdf.columns]].copy()

    if 'beginning' in gdf.columns:
        gdf['beginning'] = pd.to_datetime(gdf['beginning'], format='%Y-%m-%dT%H:%M:%S', errors='coerce').dt.year
        gdf = gdf[gdf['beginning'] > 0].copy()  # Filtramos años válidos
        gdf.rename(columns={'beginning': 'anio_construccion'}, inplace=True)
        gdf['anio_construccion'] = gdf['anio_construccion'].astype(int)
        gdf.reset_index(drop=True, inplace=True)

    print("Guardando datos del catastro en formato Parquet... \n")
    buffer = io.BytesIO()
    gdf.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    client = crear_cliente_minio()
    minio_subir_memoria(client, MINIO_CATASTRO, OBJ_CATASTRO, buffer)
    print("Datos del Catastro subidos a MinIO. \n")

if __name__ == "__main__":
    descargar_catastro()

