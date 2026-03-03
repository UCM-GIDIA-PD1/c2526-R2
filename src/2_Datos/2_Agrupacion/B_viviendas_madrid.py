import pandas as pd
import geopandas as gpd
from src.config import COMPONENTES_TRANSPORTE, MODOS
from scipy.spatial import cKDTree
import numpy as np
from src.utils.funciones_minio import bajar_minio, buscar_todos_los_archivos, crear_cliente_minio, subir_mapa_minio
from minio import Minio
import folium
import tempfile
import webbrowser

def meter_datos_transporte(gdf_viviendas, df_transporte, nombre_categoria, col_lineas='lineas', radio_metros=500):
    """
    Calcula la distancia a la parada más cercana, cuántas paradas hay a menos de X metros,
    y cuántas LÍNEAS DISTINTAS están accesibles en ese radio para cada vivienda.
    """
    
    gdf_res = gdf_viviendas.copy()

    crs_activo = gdf_viviendas.crs
    if crs_activo is None or crs_activo.to_epsg() == 4326:
        gdf_viviendas = gdf_viviendas.to_crs("EPSG:25830") 
    
    gdf_transporte = gpd.GeoDataFrame(
        df_transporte, 
        geometry=gpd.points_from_xy(df_transporte['lon'], df_transporte['lat']), 
        crs="EPSG:4326"
    ).to_crs("EPSG:25830")

    coords_viviendas = list(zip(gdf_res.geometry.x, gdf_res.geometry.y))
    coords_transporte = list(zip(gdf_transporte.geometry.x, gdf_transporte.geometry.y))

    arbol_transporte = cKDTree(coords_transporte)

    distancias_minimas, _ = arbol_transporte.query(coords_viviendas, k=1)
    gdf_res[f'dist_min_{nombre_categoria}'] = np.round(distancias_minimas, 1)

    paradas_en_radio = arbol_transporte.query_ball_point(coords_viviendas, r=radio_metros)
    
    lineas_array = df_transporte[col_lineas].values
    
    conteos_paradas = []
    conteos_lineas_unicas = []
    
    for indices_cercanos in paradas_en_radio:
        conteos_paradas.append(len(indices_cercanos))
        lineas_unicas = set()
        for idx in indices_cercanos:
            item = lineas_array[idx]
            lineas_unicas.update(item)
        
        conteos_lineas_unicas.append(len(lineas_unicas))

    gdf_res[f'{nombre_categoria}_cerca'] = conteos_paradas
    gdf_res[f'lineas_distintas_{nombre_categoria}_cerca'] = conteos_lineas_unicas

    return gdf_res

def meter_datos_secundarios(gdf_viv, df_pois, nombre_categoria, radio_metros=500):
    """
    Calcula para cada vivienda la distancia al POI (Punto de Interés) más cercano
    y cuántos POIs hay en un radio determinado (ej: 500m).
    Utiliza cKDTree para que el cálculo sea instantáneo.
    """    
    df_res = gdf_viv.copy()
    crs_activo = gdf_viv.crs
    if crs_activo is None or crs_activo.to_epsg() == 4326:
        gdf_viv = gdf_viv.to_crs("EPSG:25830") 

    gdf_poi = gpd.GeoDataFrame(
        df_pois, 
        geometry=gpd.points_from_xy(df_pois['lon'], df_pois['lat']), 
        crs="EPSG:4326"
    ).to_crs("EPSG:25830")

    coords_viviendas = list(zip(gdf_viv.geometry.x, gdf_viv.geometry.y))
    coords_pois = list(zip(gdf_poi.geometry.x, gdf_poi.geometry.y))

    arbol_pois = cKDTree(coords_pois)

    distancias_minimas, indices_cercanos = arbol_pois.query(coords_viviendas, k=1)

    pois_en_radio = arbol_pois.query_ball_point(coords_viviendas, r=radio_metros)
    
    conteos_radio = [len(lista_pois) for lista_pois in pois_en_radio]

    df_res[f'dist_min_{nombre_categoria}'] = np.round(distancias_minimas, 1)
    df_res[f'cantidad_{nombre_categoria}_cerca'] = conteos_radio

    gdf_final = gpd.GeoDataFrame(
        df_res, 
        geometry=gpd.points_from_xy(df_res['lon'], df_res['lat']), 
        crs="EPSG:4326"
    )

    return gdf_final

def limpiar_coordenadas_lejanas(df, lat_min=40.0, lat_max=41, lon_min=-3.95, lon_max=-3.35):
    """
    Filtra los puntos que caen fuera de una 'caja' lógica.
    (Las coordenadas por defecto son un recuadro amplio alrededor de Madrid).
    """
    total_antes = len(df)
    
    df_limpio = df[
        (df['lat'] >= lat_min) & (df['lat'] <= lat_max) & 
        (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
    ].copy()
        
    return df_limpio

def descargar_datos(cliente:Minio,path:str,nombre_archivo:str)->pd.DataFrame:
    df = bajar_minio(cliente,path,nombre_archivo)
    return df

def subir_viviendas_con_info(cliente:Minio,gdf:gpd.GeoDataFrame,archivo:str,path="rejillas"):
    subir_mapa_minio(cliente,gdf,path,archivo)

def visualizar_rejilla(gdf:gpd.GeoDataFrame):
    mapa_base = gdf.explore()
    folium.LayerControl().add_to(mapa_base)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        mapa_base.save(tmp.name)
        ruta_temporal = tmp.name
    webbrowser.open('file://' + ruta_temporal)

def inicio_viviendas():
    cliente = crear_cliente_minio()
    df_viviendas_venta = descargar_datos(cliente,"datos_primarios","viviendas_venta.parquet")
    df_viviendas_alquiler = descargar_datos(cliente,"datos_primarios","viviendas_alquiler.parquet")
    diccionario_viviendas = {
        "venta": df_viviendas_venta,
        "alquiler": df_viviendas_alquiler
    }
    diccionario_transporte = {}
    for transporte in COMPONENTES_TRANSPORTE:
        df = descargar_datos(cliente,"cleaned/transporte",transporte["fichero"])
        diccionario_transporte[transporte["calculo"]] = df
    datos_secundarios = buscar_todos_los_archivos(cliente,"cleaned/secundarios")
    diccionario_secundarios = {}
    for sec in datos_secundarios:
        df = descargar_datos(cliente,"cleaned/secundarios",sec)
        diccionario_secundarios[sec.removesuffix(".parquet")] = df
    for modo in MODOS:
        diccionario_viviendas[modo] = diccionario_viviendas[modo].drop(columns=["Descripcion","Url"])
        diccionario_viviendas[modo] = limpiar_coordenadas_lejanas(diccionario_viviendas[modo])
        gdf_viviendas = gpd.GeoDataFrame(diccionario_viviendas[modo],geometry=gpd.points_from_xy(diccionario_viviendas[modo]['lon'], diccionario_viviendas[modo]['lat']), 
        crs="EPSG:4326"
        ).to_crs("EPSG:25830")
        for tipo,df_s in diccionario_secundarios.items():
            gdf_viviendas = meter_datos_secundarios(gdf_viviendas,df_s,tipo)
        for tipo,df_t in diccionario_transporte.items():
            gdf_viviendas = meter_datos_transporte(gdf_viviendas,df_t,tipo)
        subir_viviendas_con_info(cliente,gdf_viviendas,f"viviendas_{modo}")
        print(f"Mapa de viviendas de {modo} subido con exito.")


if __name__ == "__main__":
    inicio_viviendas()
