import os
import requests
import pandas as pd
import io
from dotenv import load_dotenv
from src.utils.funciones_minio import crear_cliente_minio, minio_subir_memoria
'''
Script para la extracción de datos de renta media por hogar del INE, filtrado por Madrid Capital,
además selecciona el año más reciente y sube el resultado a MinIO.
'''

def descargar_ine():
    """
    Descarga datos de renta del INE vía API JSON, filtra por Madrid Capital, 
    selecciona el año más reciente y sube el resultado a MinIO.

    Raises:
        requests.exceptions.RequestException: Si la conexión con el INE falla.
    """
    print("Iniciando proceso de extracción de datos del INE... \n")
    url_ine = "https://servicios.ine.es/wstempus/js/es/DATOS_TABLA/30824?tip=AM"

    print("Descargando datos del INE... \n")
    response = requests.get(url_ine)
    if response.status_code != 200: return

    print("Procesando datos del INE... \n")
    df = pd.DataFrame(response.json())

    def sacar_codigo_seccion(metadata_lista):
        for item in metadata_lista:
            if 'Codigo' in item and len(item['Codigo']) == 10: return item['Codigo']
        return None
        
    df['CUSEC'] = df['MetaData'].apply(sacar_codigo_seccion)
    df_madrid = df[df['CUSEC'].str.startswith('28079', na=False)].copy()
    df_madrid = df_madrid[df_madrid['Nombre'].str.contains('Renta neta media por hogar', case=False)]

    df_exploded = df_madrid.explode('Data')
    df_final = pd.concat([df_exploded[['CUSEC']], df_exploded['Data'].apply(pd.Series)], axis=1)
    
    df_final['Anyo'] = pd.to_numeric(df_final['Anyo'], errors='coerce')
    df_final = df_final[df_final['Anyo'] == df_final['Anyo'].max()]

    df_final.rename(columns={'Valor': 'renta_media'}, inplace=True)
    df_exportar = df_final[['CUSEC', 'renta_media']].dropna().drop_duplicates().reset_index(drop=True)

    buffer = io.BytesIO()
    df_exportar.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    client = crear_cliente_minio()
    minio_subir_memoria(client, "datos_secundarios/ine", "renta_hogar_secciones_madrid.parquet", buffer)
    print("Renta INE subida a MinIO. \n")

if __name__ == "__main__":
    descargar_ine()