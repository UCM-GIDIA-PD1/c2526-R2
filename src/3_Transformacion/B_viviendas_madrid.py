import pandas as pd
import geopandas as gpd
from utils.config import COMPONENTES_TRANSPORTE, MODOS
from scipy.spatial import cKDTree
import numpy as np
from utils.funciones_minio import bajar_mapa_minio, bajar_minio, buscar_todos_los_archivos, crear_cliente_minio, subir_mapa_minio
from utils.config import PATH_PRIMARIOS_LIMPIO
from minio import Minio
import folium
import tempfile
import webbrowser

def meter_datos_transporte(gdf_viviendas, df_transporte, nombre_categoria, col_lineas='lineas', radio_metros=500)->gpd.GeoDataFrame:
    """
         Calcula la distancia a la parada más cercana, cuántas paradas hay a menos de X metros,
        y cuántas lineas distintas están accesibles en ese radio para cada vivienda.
    Args:
        gdf_viviendas (_type_): los puntos de las viviendas 
        df_transporte (_type_): Datos de transporte
        nombre_categoria (_type_): Tipo de transporte
        col_lineas (str, optional): Nombre de la columna que se  crea. Defaults to 'lineas'.
        radio_metros (int, optional): Radio en el que se instaura el radio de cálculo de caraterísticas. Defaults to 500.

    Returns:
        gpd.GeoDataFrame: Mapa con los cálculos realizados
    """
    
    gdf_res = gdf_viviendas.copy()
    crs_activo = gdf_viviendas.crs
    if crs_activo is None or crs_activo.to_epsg() == 4326:
        gdf_res = gdf_res.to_crs("EPSG:25830") 
    
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

def meter_datos_secundarios(gdf_viv:gpd.GeoDataFrame, df_pois:pd.DataFrame, nombre_categoria:str, radio_metros=500)->gpd.GeoDataFrame:
    """
        Calcula para cada vivienda la distancia al POI (Punto de Interés) más cercano
        y cuántos puntos hay en un radio determinado (500m).
    Args:
        gdf_viv (gpd.GeoDataFrame): mapa de las viviendas
        df_pois (pd.DataFrame): dataframe de puntos de datos secundarios
        nombre_categoria (str): nombre de la categoría de datos secundarios
        radio_metros (int, optional): Radio en el que se instaura el radio de cálculo de caraterísticas. Defaults to 500.

    Returns:
        gpd.GeoDataFrame: Mapa con datos calculados 
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

def mete_datos_catastro(gdf_viviendas:gpd.GeoDataFrame, gdf_catastro:gpd.GeoDataFrame, col_anyo='anio_construccion')->gpd.GeoDataFrame:
    """
    Cruza los puntos de las viviendas con los polígonos del catastro para 
    asignar a cada vivienda el año de construcción del edificio en el que cae.
    Args:
        gdf_viviendas (gpd.GeoDataFrame): Mapa de viviendas
        gdf_catastro (gpd.GeoDataFrame): Mapa de construcción de puntos
        col_anyo (str, optional): Nombre que se le atribuye a la columna. Defaults to 'anio_construccion'.

    Returns:
        gpd.GeoDataFrame: Mapa con anios de contrucción calculados
    """
    
    gdf_res = gdf_viviendas.copy()
    
    if gdf_res.crs != gdf_catastro.crs:
        gdf_catastro = gdf_catastro.to_crs(gdf_res.crs)
        
   
    catastro_reducido = gdf_catastro[['geometry', col_anyo]]
    
    cruce = gpd.sjoin_nearest(
        gdf_res, 
        catastro_reducido, 
        how='left', 
        max_distance= 30,
        distance_col='dist_al_edificio'
    )

    if cruce.index.duplicated().any():
        cruce = cruce[~cruce.index.duplicated(keep='first')]
    if 'index_right' in cruce.columns:
        cruce = cruce.drop(columns=['index_right'])
        
    return cruce

def mete_datos_padron(gdf_viviendas:gpd.GeoDataFrame, gdf_secciones_enriquecidas:gpd.GeoDataFrame)->gpd.GeoDataFrame:
    """
    Cruza espacialmente las viviendas con el mapa de secciones censales ya procesado,
    transfiriendo directamente la demografía (población total) a cada piso.
    """
    gdf_res = gdf_viviendas.copy()
    
    if gdf_res.crs != gdf_secciones_enriquecidas.crs:
        gdf_secciones_enriquecidas = gdf_secciones_enriquecidas.to_crs(gdf_res.crs)

    columnas_a_transferir = [
        'geometry', 'poblacion_total', 'pct_extranjeros', 
        #pct_espanoles no lo pasamos pq tendria correacioon perfecta con pct_extranjeros, y el modelo no ganaria nada con esa info extra  
        'pct_mayores_65', 'pct_jovenes_30'
    ]
    
    cols_existentes = [c for c in columnas_a_transferir if c in gdf_secciones_enriquecidas.columns]
    gdf_mapa_reducido = gdf_secciones_enriquecidas[cols_existentes]
    
    cruce = gpd.sjoin(
        gdf_res, 
        gdf_mapa_reducido, 
        how='left', 
        predicate='within'
    )
    
    if 'index_right' in cruce.columns:
        cruce = cruce.drop(columns=['index_right'])
        
    return cruce

def limpiar_coordenadas_lejanas(df, lat_min=40.28, lat_max=40.65, lon_min=-3.83, lon_max=-3.48):
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
    """
        Descarga los datos que se desea completar
    Args:
        cliente (Minio): Cliente de Minio
        path (str): Carpeta donde se encuentra el archivo 
        nombre_archivo (str): Nombre del archivo que se descarga

    Returns:
        pd.DataFrame: Dataframe de los datos descargados
    """
    df = bajar_minio(cliente,path,nombre_archivo)
    return df

def subir_viviendas_con_info(cliente:Minio,gdf:gpd.GeoDataFrame,archivo:str,path="rejillas"):
    """
        Sube los datos completados a la carpeta rejillas
    Args:
        cliente (Minio): Cliente de minio
        gdf (gpd.GeoDataFrame): datos completados
        archivo (str): nombre del archivo
        path (str, optional): Carpeta donde se almacena. Defaults to "rejillas".
    """
    subir_mapa_minio(cliente,gdf,path,archivo)

def visualizar_rejilla(gdf:gpd.GeoDataFrame):
    """
        Función auxiliar para visualizar rejillas en un html
    Args:
        gdf (gpd.GeoDataFrame): Rejilla que se desea visualizar
    """
    mapa_base = gdf.explore()
    folium.LayerControl().add_to(mapa_base)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        mapa_base.save(tmp.name)
        ruta_temporal = tmp.name
    webbrowser.open('file://' + ruta_temporal)

def inicio_viviendas():
    """
    Funcion main de esta parte del pipeline de ejecución 
    Se encarga de descargar todos los datasets y subirlos a rejillas
    """
    cliente = crear_cliente_minio()
    df_viviendas_venta = descargar_datos(cliente,PATH_PRIMARIOS_LIMPIO,"viviendas_venta.parquet")
    df_viviendas_alquiler = descargar_datos(cliente,PATH_PRIMARIOS_LIMPIO,"viviendas_alquiler.parquet")
    diccionario_viviendas = {
        "venta": df_viviendas_venta,
        "alquiler": df_viviendas_alquiler
    }
    gdf_catastro = bajar_mapa_minio(cliente,"cleaned/catastro","anio_construccion")
    gdf_padron_secciones = bajar_mapa_minio(cliente, "rejillas", "secciones censales")
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
        gdf_viviendas = mete_datos_catastro(gdf_viviendas,gdf_catastro)
        gdf_viviendas = mete_datos_padron(gdf_viviendas, gdf_padron_secciones)
        subir_viviendas_con_info(cliente,gdf_viviendas,f"viviendas_{modo}")
        print(f"Mapa de viviendas de {modo} subido con exito.")


if __name__ == "__main__":
    inicio_viviendas()
