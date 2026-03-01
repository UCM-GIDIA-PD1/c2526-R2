import geopandas as gpd
import matplotlib.pyplot as plt
import time
import folium
import matplotlib
import mapclassify
from src.config import TIPOS_REJILLAS,MINIO_REJILLAS_SUCIO,PATH_PRIMARIOS_LIMPIO,ARCHIVOS_COORDENADAS
from src.utils.funciones_minio import bajar_mapa_minio,crear_cliente_minio,bajar_minio
import pandas as pd
from minio import Minio

def descarga_rejilla(tipo:str,cliente:Minio):
    gdf = bajar_mapa_minio(cliente,MINIO_REJILLAS_SUCIO,f"{tipo.replace(' ','_')}_madrid")
    return gdf

def obtener_coordenadas_procesadas(client:Minio)->pd.DataFrame:
    path = PATH_PRIMARIOS_LIMPIO
    archivo = ARCHIVOS_COORDENADAS
    try:
        print(f" Buscando memoria de coordenadas en: {path}/¨{archivo}")
        df_coordenadas = bajar_minio(client,path,archivo)
        print(f" Cargadas {len(df_coordenadas)} calles conocidas.")
        return df_coordenadas
    except Exception as e:
        print(f" Error al procesar la memoria: {e}")
        return pd.DataFrame()



def inicio_rejillas():
    print("Agrupar datos a las rejillas:")
    cliente = crear_cliente_minio()
    df_coorenadas = obtener_coordenadas_procesadas(cliente)
    for rejilla in TIPOS_REJILLAS:
        gdf_rejilla = descarga_rejilla(rejilla,cliente)


