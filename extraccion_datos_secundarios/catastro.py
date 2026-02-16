import os
import pandas as pd
import requests
import zipfile
import geopandas as gpd
import pyarrow as pa 
from pathlib import Path
from minio import Minio
from dotenv import load_dotenv

ruta_original = Path(__file__).resolve().parent
ruta_datos = ruta_original / 'datos_secundarios' / 'catastro_madrid'
ruta_datos.mkdir(parents=True, exist_ok=True)

url_catastro= "https://www.catastro.hacienda.gob.es/INSPIRE/Buildings/28/28900-MADRID/A.ES.SDGC.BU.28900.zip"
nombre_zip = ruta_datos / "catastro_madrid.zip"
nombre_parquet = ruta_datos / "edificios_madrid.parquet"

def descargar_y_procesar():
    response = requests.get(url_catastro)
    
    if response.status_code == 200:
        with open(nombre_zip, 'wb') as f:
            f.write(response.content)
        
        # Extracción y filtrado
        with zipfile.ZipFile(nombre_zip, 'r') as z:
            archivo_gml = [f for f in z.namelist() if f.endswith('building.gml')][0]
            z.extract(archivo_gml, path=ruta_datos)
            ruta_archivo = ruta_datos / archivo_gml

        archivo = gpd.read_file(ruta_archivo)
        
        # Filtramos columnas
        columnas_utiles = ['geometry', 'beginning']
        archivo_final = archivo[[c for c in columnas_utiles if c in archivo.columns]].copy()
        
        # Nos quedamos solo con el año de construcción y borramos los q no tengan esa info
        archivo_final['beginning'] = pd.to_datetime(archivo_final['beginning'], format='%Y-%m-%dT%H:%M:%S', errors='coerce').dt.year
        archivo_final = archivo_final[archivo_final['beginning'] > 0].copy()

        # Guardamos en Parquet
        archivo_final.to_parquet(nombre_parquet)
        
        # Borramos los otros archivos
        if ruta_archivo.exists(): ruta_archivo.unlink() # Borra el .gml
        ruta_gfs = ruta_archivo.with_suffix('.gfs')
        if ruta_gfs.exists(): ruta_gfs.unlink() #Borra el .gfs
        if nombre_zip.exists(): nombre_zip.unlink()   # Borra el .zip

        return nombre_parquet
    else:
        print(f"ERROR: {response.status_code}")
        return None
    
def subir_a_minio(ruta_archivo):
    if not ruta_archivo: return
    
    load_dotenv()
    minio_bucket = os.getenv("MINIO_BUCKET")
    minio_group = os.getenv("GROUP_PATH")
    path_subcarpeta = "datos_secundarios/catastro"

    client = Minio(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
    )

    nombre_objeto = f"{minio_group}/{path_subcarpeta}/{ruta_archivo.name}"
    
    client.fput_object(
        bucket_name=minio_bucket,
        object_name=nombre_objeto,
        file_path=str(ruta_archivo)
    )
    
    # Limpiamos los archivos locales
    ruta_archivo.unlink() # Borra el parquet
    if ruta_datos.exists():
        # Borra la carpeta temporal si está vacía
        try:
            ruta_datos.rmdir()
            (ruta_original / 'datos_secundarios').rmdir()
        except:
            pass 

if __name__ == "__main__":
    ruta_parquet = descargar_y_procesar()
    subir_a_minio(ruta_parquet)
    