import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from src.config import PAIS,CIUDAD
import re
from src.utils.funciones_minio import *
from tqdm import tqdm
from PIL import Image
import numpy as np
import time

columnas_características = ['id','Nombre','Barrio','Distrito','Calle','Precio','Superficie','Num_habitaciones','Banyos','Planta',
                      'Ventanas','Ascensor','Terraza','Balcon','Equipamiento','Cocina','Orientacion','Consumo','Descripcion','Anuncia','Url']

columnas_imagenes = ['id','Imagenes']
modos = ['venta','alquiler']
path_limpio = "datos_primarios"
path_raw = "raw/datos_primarios/"
archivo_coordenadas = "coordenadas.parquet"
archivos_viviendas = "viviendas"
archivo_imagenes = "imagenes"

def limpia_direccion(direccion:str):
    pais = PAIS
    ciudad = CIUDAD

    dir_limpia = direccion.upper()

    abreviaturas= {
        r'\bPS\b': 'PASEO',
        r'\bC/\b': 'CALLE',
        r'\bC\.\b': 'CALLE',
        r'\bCl\\b':'CALLE',
        r'\bC\b': 'CALLE',     
        r'\bAV\b': 'AVENIDA',
        r'\bAVDA\b': 'AVENIDA',
        r'\bPL\b': 'PLAZA',
        r'\bCTRA\b': 'CARRETERA'
    }

    for abreviatura,sustitucion in abreviaturas.items():
        dir_limpia = re.sub(abreviatura, sustitucion, dir_limpia)
    
    # se detectaron casos en los que idealista pone (s/n) para indicar que no se ha indicado el numero
    dir_limpia = re.sub(r'\s*S/N-?\s*', ' ', dir_limpia)

    dir_limpia = dir_limpia.split(' - ')[0]

    partes = dir_limpia.split(',')
    if len(partes) > 2:
        dir_limpia = f"{partes[0].strip()}, {partes[1].strip()}"

    dir_limpia = re.sub(r',?\s*\d+[ºª].*$', '', dir_limpia)

    dir_limpia = re.sub(r',?\s*\d+-\d+\s*$', '', dir_limpia)

    dir_limpia = re.sub(r'\s+', ' ', dir_limpia).strip()
    dir_limpia = re.sub(r',$', '', dir_limpia)

    return f"{dir_limpia}, {ciudad}, {pais}"


def descargar_anuncios(client:Minio,modo:str)->pd.DataFrame:
    path = f'{path_raw}{modo}'
    parquets = buscar_todos_los_archivos(client,path)
    ignorar = {'ids.parquet'}
    df_res = pd.DataFrame()
    for parquet in tqdm(parquets,desc = "Descargando totalidad de anuncios"):
        if parquet not in ignorar:
            df = bajar_minio_especifico(client,path,parquet.replace(path,''),columnas=columnas_características)
            df_res = pd.concat([df_res,df],ignore_index=True)
    
    return df_res

def subir_viviendas_limpio(df:pd.DataFrame,cliente:Minio,modo:str):
    path = path_limpio
    archivo = f"{archivos_viviendas}_{modo}.parquet"   
    subir_minio(df,cliente,path,archivo)

def sustituir_valores_nulos(df:pd.DataFrame)->pd.DataFrame:

    df_limpiado = df.copy()

    df_limpiado['Planta'] = df_limpiado['Planta'].fillna(0)

    df_limpiado["Precio"] = df_limpiado["Precio"].astype(str).str.replace('.','')
    df_limpiado["Precio"] = pd.to_numeric(df_limpiado["Precio"],error='coerce')

    df_limpiado["Orientacion"] = (df_limpiado["Orientacion"].astype(str).str.replace(',','',regex = False).str.strip().str.capitalize())

    columnas_nulas = ['ventanas', 'orientacion']

    for col in columnas_nulas:
        if col in df_limpiado.columns:
            df_limpiado[col] = df_limpiado[col].astype(str).str.replace(r'(?i)(no determinado|no indicado)','No determinado', regex=True)

            df_limpiado[col] = df_limpiado[col].replace('nan', 'No determinado')
            df_limpiado[col] = df_limpiado[col].replace('Nan', 'No determinado')

    return df_limpiado


def obtener_coordenadas_procesadas(client:Minio)->pd.DataFrame:
    path = path_limpio
    archivo = archivo_coordenadas
    try:
        print(f" Buscando memoria de coordenadas en: {path}/¨{archivo}")
        df_coordenadas = bajar_minio(client,path_limpio,archivo)
        print(f" Cargadas {len(df_coordenadas)} calles conocidas.")
        return df_coordenadas
    except Exception as e:
        print(f" Error al procesar la memoria: {e}")
        return pd.DataFrame()

def subir_coordenadas(client:Minio,df_coordenadas:pd.DataFrame)->None:
    path = path_limpio
    archivo = archivo_coordenadas
    subir_minio(df_coordenadas,client,path,archivo)

def descargar_imagenes(cliente:Minio,path:str,nombre_archivo:str)->pd.DataFrame:
    df = bajar_minio_especifico(cliente,path,nombre_archivo,columnas_imagenes)
    return df

def separar_imagenes(cliente:Minio):
    for modo in modos:
        num_archivo = 1
        path_sucio = f"{path_raw}{modo}"
        parquets = buscar_todos_los_archivos(cliente,path_sucio)
        df_buffer = pd.DataFrame()
        for parquet in tqdm(parquets,desc="Transfiriendo imagenes a limpio..."):
            df_buffer = pd.concat([df_buffer,descargar_imagenes(cliente,path_sucio,parquet)])
            if len(df_buffer)>600:
                df_subir = df_buffer.iloc[:600].copy()
                subir_minio(df_subir,cliente,path_limpio,f"{archivo_imagenes}_n_{num_archivo}")
                num_archivo+=1
            


def aportar_coordenadas(df_venta,df_alquiler,cliente:Minio):
    """
    Toma un DataFrame, limpia la columna de direcciones y añade columnas de Latitud, Longitud y Tipo.
    """

    tqdm.pandas(desc=" Geocodificando pisos")

    geolocator = Nominatim(user_agent="maiday_bot_v1")
    
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

    def procesar_fila(direccion_sucia):
        dir_limpia = limpia_direccion(direccion_sucia)
            
        if not dir_limpia:
            return pd.Series([None, None, None, None])
                
        try:
            loc = geocode(dir_limpia, addressdetails=True)
                
            if loc:
                lat = loc.latitude
                lon = loc.longitude
                    
                # Opción A: Lo que nos dice Nominatim (ej: 'residential', 'pedestrian', 'secondary')
                tipo_nominatim = loc.raw.get('type', 'desconocido')
                    
                # Opción B: El tipo real español que limpiamos (ej: 'CALLE', 'PASEO')
                tipo_espanol = dir_limpia.split(' ')[0]
                return pd.Series([lat, lon, tipo_nominatim, tipo_espanol])
            else:
                return pd.Series([None, None, "No Encontrado", None])
                    
        except Exception as e:
            return pd.Series([None, None, "Error API", None])

    df_alquiler['Direccion'] = df_alquiler['Calle'].apply(lambda x: limpia_direccion(x))
    df_venta['Direccion'] = df_venta['Calle'].apply(lambda x: limpia_direccion(x))
    df_coordenadas = obtener_coordenadas_procesadas(cliente)

    direcciones_alquiler = df_alquiler[["Direccion"]].dropna()
    direcciones_venta = df_venta[["Direccion"]].dropna()

    df_unicas = pd.concat([direcciones_alquiler,direcciones_venta]).drop_duplicates().copy()

    if not df_coordenadas.empty:
        calles_conocidas = df_coordenadas["Direccion"].tolist()
        df_unicas = df_unicas[~df_unicas["Direccion"].isin(calles_conocidas)].copy()

    print(f" Iniciando geocodificación de {len(df_unicas)} anuncios...")
    df_unicas[['Latitud', 'Longitud', 'Tipo_OSM', 'Tipo_Via']] = df_unicas["Direccion"].progress_apply(procesar_fila)
    
    df_res = pd.concat([df_unicas,df_coordenadas],ignore_index=True)
    subir_coordenadas(cliente,df_res)
    print(" Geocodificación terminada.")

    return df_res

def limpiar_memoria_raw():
    cliente = crear_cliente_minio()
    df_alquiler = descargar_anuncios(cliente,"alquiler")
    df_venta = descargar_anuncios(cliente,"venta")
    df_coordenadas = aportar_coordenadas(df_venta,df_alquiler,cliente)
    df_venta = pd.merge(df_venta,df_coordenadas,on = "Direccion",how = 'left')
    df_alquiler = pd.merge(df_alquiler,df_coordenadas,on = "Direccion",how = 'left')
    df_venta = sustituir_valores_nulos(df_venta)
    df_alquiler = sustituir_valores_nulos(df_alquiler)
    subir_viviendas_limpio(df_alquiler,cliente,"alquiler")
    subir_viviendas_limpio(df_venta,cliente,"venta")

    print("Limpieza de datasets de viviendas completada.") 



if __name__=="__main__":
    cliente = crear_cliente_minio()
    df = descargar_imagenes(cliente,f"{path_raw}alquiler","batch_moratalaz_n_1.parquet")