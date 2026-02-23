import os
import requests
import pandas as pd
import io
from dotenv import load_dotenv
from funciones_minio import crear_cliente_minio, minio_subir_memoria


def sacar_codigo_seccion(metadata_lista):
    """
    Extrae el código de sección censal (10 dígitos) desde la lista de metadatos.

    Args:
        metadata_lista (list): Lista de diccionarios con información de metadatos.

    Returns:
        str | None: Código de sección encontrado o None si no existe.
    """
    try:
        for item in metadata_lista:
            # El código de sección tiene 10 dígitos, por ejemplo: 2807901001
            # 28 Madrid, 079 Madrid Capital, 01 Distrito (Centro), 001 Sección
            if 'Codigo' in item and len(item['Codigo']) == 10: 
                return item['Codigo']
    except:
        pass
    return None


def descargar_ine():
    """
    Descarga datos del INE, filtra la renta neta media por hogar
    para las secciones censales de Madrid en el último año disponible
    y sube el resultado en formato Parquet a MinIO.

    Returns:
        Nada: Se sube todo al MinIO directamente
    """
    print("Iniciando proceso de extracción del INE...")
    url_ine = "https://servicios.ine.es/wstempus/js/es/DATOS_TABLA/30824?tip=AM"
    
    print("Descargando datos del INE...")
    response = requests.get(url_ine)
    
    if response.status_code != 200:
        print(f"ERROR: {response.status_code}")
        return None
    
    print("Procesando datos del INE...")
    df = pd.DataFrame(response.json())
        
    df['CUSEC'] = df['MetaData'].apply(sacar_codigo_seccion)
    df_madrid = df[df['CUSEC'].str.startswith('28079', na=False)].copy()
        
    # Buscamos solo las filas de Renta neta media por hogar
    filtro_renta = df_madrid['Nombre'].str.contains('Renta neta media por hogar', case=False, na=False)
    df_madrid = df_madrid[filtro_renta]

    # Convertimos cada elemento (año) de la lista 'Data' en una fila separada
    df_exploded = df_madrid.explode('Data')
    datos_extraidos = df_exploded['Data'].apply(pd.Series)
    df_final = pd.concat([df_exploded[['CUSEC']], datos_extraidos], axis=1)
        
    # Filtramos por el último año disponible
    df_final['Anyo'] = pd.to_numeric(df_final['Anyo'], errors='coerce')
    ultimo_ano = df_final['Anyo'].max()
    df_final = df_final[df_final['Anyo'] == ultimo_ano]

    df_final.rename(columns={'Valor': 'renta_media'}, inplace=True)
    df_exportar = df_final[['CUSEC', 'renta_media']].dropna().drop_duplicates()
    df_exportar.reset_index(drop=True, inplace=True)

    # Guardamos en Parquet y subimos a minIO
    print("Guardando datos del INE en formato Parquet...")
    buffer = io.BytesIO()
    df_exportar.to_parquet(buffer, index=False)
    buffer.seek(0)

    load_dotenv()
    cliente = crear_cliente_minio()
    nombre_objeto = f"{os.getenv('MINIO_GROUP_PATH')}/datos_secundarios/ine/renta_hogar_secciones_madrid.parquet"
    minio_subir_memoria(cliente, buffer, nombre_objeto)
    print("Archivo del INE subido a MinIO")


if __name__ == "__main__":
    descargar_ine()