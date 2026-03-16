import pandas as pd
import numpy as np
import requests
import io
from utils.funciones_minio import crear_cliente_minio, minio_subir_memoria
from utils.config import URL_PADRON, MINIO_PADRON, OBJ_PADRON_BAR, OBJ_PADRON_SEC

'''
Script para la extracción de datos del padrón municipal de Madrid.
Calcula la población total, métricas demográficas (edad) y nacionalidad
por barrio y sección censal, y sube los resultados a MinIO.
'''

def agrupar_y_calcular_metricas(df, columna_id):
    """
    Función auxiliar que agrupa el DataFrame por la columna indicada (Barrio o Sección) 
    y calcula la población total, rangos de edad y procedencia (españoles/extranjeros).
    """
    # Sumamos las métricas absolutas
    df_agrupado = df.groupby(columna_id).agg(
        poblacion_total=('poblacion_total', 'sum'),
        pob_mayores_65=('pob_mayores_65', 'sum'),
        pob_jovenes_30=('pob_jovenes_30', 'sum'),
        pob_espanoles=('pob_espanoles', 'sum'),
        pob_extranjeros=('pob_extranjeros', 'sum')
    ).reset_index()

    # Calculamos los porcentajes y redondeamos a 2 decimales
    df_agrupado['pct_mayores_65'] = np.where(
        df_agrupado['poblacion_total'] > 0, 
        (df_agrupado['pob_mayores_65'] / df_agrupado['poblacion_total']) * 100, 0
    ).round(2)
    
    df_agrupado['pct_jovenes_30'] = np.where(
        df_agrupado['poblacion_total'] > 0, 
        (df_agrupado['pob_jovenes_30'] / df_agrupado['poblacion_total']) * 100, 0
    ).round(2)

    df_agrupado['pct_espanoles'] = np.where(
        df_agrupado['poblacion_total'] > 0, 
        (df_agrupado['pob_espanoles'] / df_agrupado['poblacion_total']) * 100, 0
    ).round(2)

    df_agrupado['pct_extranjeros'] = np.where(
        df_agrupado['poblacion_total'] > 0, 
        (df_agrupado['pob_extranjeros'] / df_agrupado['poblacion_total']) * 100, 0
    ).round(2)

    return df_agrupado


def descargar_padron():
    print("Descargando datos del Padrón... \n")
    response = requests.get(URL_PADRON)
    
    if response.status_code != 200: 
        print(f"ERROR: {response.status_code}")
        return

    print("Procesando datos del padrón y calculando métricas... \n")
    df = pd.DataFrame(response.json())

    # 1. Población Total, Españoles y Extranjeros
    df['pob_espanoles'] = df['ESPANOLESHOMBRES'] + df['ESPANOLESMUJERES']
    df['pob_extranjeros'] = df['EXTRANJEROSHOMBRES'] + df['EXTRANJEROSMUJERES']
    df['poblacion_total'] = df['pob_espanoles'] + df['pob_extranjeros']

    # 2. Edades (mayores de 65 y menores de 30)
    df['EDAD'] = df['COD_EDAD_INT'].astype(str).str.extract(r'(\d+)').astype(float)
    df['pob_mayores_65'] = np.where(df['EDAD'] >= 65, df['poblacion_total'], 0)
    df['pob_jovenes_30'] = np.where(df['EDAD'] < 30, df['poblacion_total'], 0)

    # 3. Crear identificadores de Barrio y Sección Censal
    df['COD_BAR'] = df['COD_DISTRITO'].astype('Int64').astype(str).str.zfill(2) + df['COD_BARRIO'].astype('Int64').astype(str)
    df['CUSEC'] = '28079' + df['COD_DIST_SECCION'].astype('Int64').astype(str).str.zfill(5)

    print("Agrupando por Barrios y Secciones Censales... \n")
    df_barrios = agrupar_y_calcular_metricas(df, 'COD_BAR')
    df_secciones = agrupar_y_calcular_metricas(df, 'CUSEC')

    df_barrios = df_barrios[['COD_BAR', 'poblacion_total', 'pct_espanoles', 'pct_extranjeros', 'pct_mayores_65', 'pct_jovenes_30']]
    df_secciones = df_secciones[['CUSEC', 'poblacion_total', 'pct_espanoles', 'pct_extranjeros', 'pct_mayores_65', 'pct_jovenes_30']]

    print("Subiendo datos procesados del padrón a MinIO... \n")
    client = crear_cliente_minio()

    archivos_a_subir = [
        (df_barrios, OBJ_PADRON_BAR, "Barrios INE procesados y subidos a MinIO."),
        (df_secciones, OBJ_PADRON_SEC, "Secciones INE procesadas y subidas a MinIO.")
    ]

    for df_upload, obj_name, mensaje in archivos_a_subir:
        buffer = io.BytesIO()
        df_upload.to_parquet(buffer, index=False)
        buffer.seek(0)
        minio_subir_memoria(client, MINIO_PADRON, obj_name, buffer)
        print(mensaje)


if __name__ == "__main__":
    descargar_padron()