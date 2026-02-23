import os
import requests
import geopandas as gdp
import io
from dotenv import load_dotenv
from funciones_minio import crear_cliente_minio, minio_subir_memoria

def descargar_secciones_censales():
    print("Iniciando proceso de extracción de las secciones censales del Ayuntamiento de Madrid...")
    url = "https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Seccionado/TopoJSON/Secciones_Censales.json"
    
    print("Descargando datos de las secciones censales...")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"ERROR: {response.status_code}")
        return None
    
    print("Procesando datos de las secciones censales...")
    gdf = gdp.read_file(io.BytesIO(response.content))
    
    # Nos quedamos solo con el código de sección y la geometría
    gdf['CUSEC'] = "28079" + gdf['COD_SECCIO'].astype(str).str.zfill(5)
    gdf_final = gdf[['geometry', 'CUSEC']].copy()
    
    # Guardamos en Parquet
    buffer = io.BytesIO()
    gdf_final.to_parquet(buffer, index=False)
    buffer.seek(0)

    load_dotenv()
    cliente = crear_cliente_minio()
    nombre_objeto = f"{os.getenv('MINIO_GROUP_PATH')}/datos_secundarios/secciones_censales/secciones_censales_madrid.parquet"
    
    minio_subir_memoria(cliente, buffer, nombre_objeto)
    print("Archivo de secciones censales subido a MinIO")

if __name__ == "__main__":
    descargar_secciones_censales()