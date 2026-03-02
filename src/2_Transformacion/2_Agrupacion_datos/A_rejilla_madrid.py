import geopandas as gpd
import matplotlib.pyplot as plt
import time
import folium
import matplotlib
import mapclassify
from src.config import TIPOS_REJILLAS,MINIO_REJILLAS_SUCIO,PATH_PRIMARIOS_LIMPIO,ARCHIVOS_COORDENADAS,MODOS,ARCHIVOS_VIVIENDAS
from src.utils.funciones_minio import bajar_mapa_minio,crear_cliente_minio,bajar_minio
import pandas as pd
from minio import Minio

def descarga_rejilla(tipo:str,cliente:Minio):
    gdf = bajar_mapa_minio(cliente,MINIO_REJILLAS_SUCIO,f"{tipo.replace(' ','_')}_madrid")
    return gdf

def extraer_mapa_principal(df_puntos, gdf_mapa_completo, id_columna, lat_col='lat', lon_col='lon'):
    """
    Toma tus puntos, mira qué polígonos del mapa oficial tocan, 
    y te devuelve un GeoDataFrame limpio solo con esos polígonos.
    Ideal para Barrios y Secciones Censales.
    """
    print(f" Generando máscara neutra para '{id_columna}'...")
    
    gdf_puntos = gpd.GeoDataFrame(
        df_puntos, 
        geometry=gpd.points_from_xy(df_puntos[lon_col], df_puntos[lat_col]),
        crs="EPSG:4326"
    )

    gdf_puntos = gdf_puntos.to_crs(gdf_mapa_completo.crs)
    
    cruce = gpd.sjoin(gdf_puntos, gdf_mapa_completo[[id_columna, 'geometry']], how='inner', predicate='within')
    
    ids_validos = cruce[id_columna].unique()
    
    gdf_mascara = gdf_mapa_completo[gdf_mapa_completo[id_columna].isin(ids_validos)]
    gdf_mascara_limpia = gdf_mascara.copy()
    
    print(f" Máscara creada: {len(gdf_mascara_limpia)} polígonos retenidos.")
    return gdf_mascara_limpia


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

def mete_datos_viviendas(gdf:gpd.GeoDataFrame,cod_rejilla:str,df_viviendas:pd.DataFrame,tipo:str)->gpd.GeoDataFrame:
    """
    Toma una rejilla (barrios, h3) y un df de viviendas.
    Asigna una categoria a cada vivienda por Lat/Lon.
    Luego calcula medias, proporciones y devuelve el mapa listo.
    """

    df_viv = df_viviendas.copy()
    gdf_res = gdf.copy()

    gdf_viv = gpd.GeoDataFrame(df_viv,geometry=gpd.points_from_xy(df_viv["lon"],df_viv["lat"],crs = "EPSG:4326"))
    gdf_viv = gdf_viv.to_crs(gdf_res.crs)

    gdf_conjunto = gpd.sjoin(gdf_viv,gdf_res[[cod_rejilla,"geometry"]],how = 'inner',predicate='within')

    df_viv = pd.DataFrame(gdf_conjunto).drop(columns = ['geometry','index_right'])

    df_viv['Precio_m2'] = df_viv['Precio'] / df_viv['Superficie']

    cols_boleanos = ['Ascensor','Terraza','Balcon','Equipamiento','Cocina']

    for col in cols_boleanos:
        if col in df_viv.columns:
            df_viv[col] = df_viv[col].astype(float)

    agrupaciones = {
        'Precio':'mean',
        'Superficie':'mean',
        'Precio_m2':'mean',
        cod_rejilla:'size'
    }

    for col in cols_boleanos:
        if col in df_viv.columns:
            agrupaciones[col] = 'mean'

    df_agrupado = df_viv.groupby(cod_rejilla).agg(agrupaciones)

    nuevos_nombres = {
        'Precio':f'Media_precio_{tipo}',
        'Superficie':f'Media_superficie_{tipo}',
        'Precio_m2':f'Media_precio_m2_{tipo}',
        cod_rejilla:f'Num_viviendas_{tipo}'
    }

    for col in cols_boleanos:
        nuevos_nombres[col] = f"Prop_{col.lower()}_{tipo}"

    df_agrupado = df_agrupado.rename(columns=nuevos_nombres).reset_index()

    gdf_res = gdf_res.merge(df_agrupado, on=cod_rejilla, how='left')

    gdf_res[f'Num_viviendas_{tipo}'] = gdf_res[f'Num_viviendas_{tipo}'].fillna(0)
    gdf_res[f'Prop_viviendas_{tipo}'] = round(gdf_res[f'Num_viviendas_{tipo}'] / gdf_res['AREA'], 2)

    for c in gdf_res.columns:
        gdf_res[c] = gdf_res.fillna(0)

    print(f"Datos de {tipo} fusionados con la rejilla")
    return gdf_res

def descargar_viviendas(cliente:Minio,modo:str)->pd.DataFrame:
    df = bajar_minio(cliente,f"{PATH_PRIMARIOS_LIMPIO}/{modo}",ARCHIVOS_VIVIENDAS)
    return df

def calcula_area(gdf:gpd.GeoDataFrame):
    crs_activo = gdf.crs
    if crs_activo is None or crs_activo.to_epsg() == 4326:
        gdf_res = gdf_res.to_crs("EPSG:25830") 
        
    gdf['AREA'] = round(gdf["AREA"] / 1000000,5)
    
    if crs_activo is not None:
        gdf_res = gdf_res.to_crs(crs_activo)

def mete_datos_mapa(gdf:gpd.GeoDataFrame,cliente:Minio)->gpd.GeoDataFrame:
    for sector in MODOS:
        df = descargar_viviendas(cliente,sector)


def inicio_rejillas():
    print("Agrupar datos a las rejillas:")
    cliente = crear_cliente_minio()
    df_coorenadas = obtener_coordenadas_procesadas(cliente)
    for rejilla in TIPOS_REJILLAS:
        gdf_rejilla = descarga_rejilla(rejilla["tipo"],cliente)
        gdf_rejilla = extraer_mapa_principal(df_coorenadas,gdf_rejilla,rejilla["columna_id"])




