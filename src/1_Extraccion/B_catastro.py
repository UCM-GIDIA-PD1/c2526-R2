import os
import pandas as pd
import requests
import zipfile
import geopandas as gpd
import io
from dotenv import load_dotenv
from funciones_minio import crear_cliente_minio, minio_subir_memoria

def descargar_catastro():
    print("Iniciando proceso de extracción del catastro...")
    url_catastro= "https://www.catastro.hacienda.gob.es/INSPIRE/Buildings/28/28900-MADRID/A.ES.SDGC.BU.28900.zip"

    print("Descargando datos del catastro...")
    response = requests.get(url_catastro)
    zip_buffer = io.BytesIO(response.content)
    
    if response.status_code != 200:
        print(f"ERROR: {response.status_code}")
        return None
    
    print("Procesando datos del catastro...")
    with zipfile.ZipFile(zip_buffer, 'r') as z:
        archivo_gml = [f for f in z.namelist() if f.endswith('building.gml')][0]
        with z.open(archivo_gml) as gml_file:
            gdf = gpd.read_file(gml_file)

    print("Sacando año de construcción...")
    columnas_utiles = ['geometry', 'beginning']
    gdf = gdf[[c for c in columnas_utiles if c in gdf.columns]].copy()

    if 'beginning' in gdf.columns:
        gdf['beginning'] = pd.to_datetime(gdf['beginning'], format='%Y-%m-%dT%H:%M:%S', errors='coerce').dt.year
        gdf = gdf[gdf['beginning'] > 0].copy()  # Filtramos años válidos
        gdf.rename(columns={'beginning': 'anio_construccion'}, inplace=True)
        gdf.reset_index(drop=True, inplace=True)
    
    print("Guardando datos del catastro en formato Parquet...")
    buffer = io.BytesIO()  
    gdf.to_parquet(buffer, index=False)
    buffer.seek(0)

    load_dotenv()
    cliente = crear_cliente_minio()
    nombre_objeto = f"{os.getenv('MINIO_GROUP_PATH')}/datos_secundarios/catastro/edificios_madrid.parquet"
    
    minio_subir_memoria(cliente, buffer, nombre_objeto)
    print("Archivo del catastro subido a MinIO")

if __name__ == "__main__":
    descargar_catastro()