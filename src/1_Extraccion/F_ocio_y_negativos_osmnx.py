import os
import osmnx as ox
import pandas as pd
import io
from src.utils.funciones_minio import crear_cliente_minio, minio_subir_memoria

'''
Script para la extracción de puntos de interés desde OpenStreetMap (OSM),
enfocado en indicadores de ocio (comercios, deportes, cultura, vida nocturna) y negativos (industrias, vertederos, prisiones).
'''

def procesar_y_subir_osm(tags: dict, nombre_fichero: str, subcarpeta: str):
    """
    Extrae geometrías de OSM, calcula sus centroides en coordenadas geográficas,
    genera un archivo Parquet y lo sube al servidor MinIO.

    Args:
        tags (dict): Diccionario de etiquetas de OSM a extraer (amenities, leisure, etc).
        nombre_fichero (str): Nombre final del archivo .parquet.
        subcarpeta (str): Carpeta destino dentro del bucket (ocio o negativos).

    Raises:
        Exception: Si no se encuentran elementos para los tags proporcionados en el área.
    """
    print(f"Iniciando extracción de datos OSM para {nombre_fichero}...\n")
    place = "Madrid, Spain"
    ox.settings.use_cache = True
    
    # Extracción
    features = ox.features.features_from_place(place, tags=tags)
    
    if features.empty:
        print(f"No se encontraron datos para {nombre_fichero} \n")
        return

    # Calculamos el centroide (Para los sitios grandes q tengan mas de una coord)
    # EPSG:25830 es el sistema de coordenadas oficial para España peninsular
    # EPSG:4326 es el sistema de latitud/longitud estándar
    # Requiere pasar de lat/lon a metros para calcular el centroide correctamente, y luego volver a lat/lon

    print(f"Calculando centroides para {nombre_fichero}... \n")
    features_proj = features.to_crs(epsg=25830)
    features['lat'] = features_proj.centroid.to_crs(epsg=4326).y
    features['lon'] = features_proj.centroid.to_crs(epsg=4326).x

    # Clasificación y limpieza
    print(f"Clasificando y limpiando datos para {nombre_fichero}... \n")
    if 'ocio' in nombre_fichero:
        features['tipo'] = features.apply(identificar_tipo_ocio, axis=1)
    else:
        features['tipo'] = features.apply(identificar_tipo_negativo, axis=1)

    columnas = ['name', 'lat', 'lon', 'tipo']
    df_final = features[[c for c in columnas if c in features.columns]].copy()
    df_final['name'] = df_final['name'].fillna(df_final['tipo'])
    df_final = df_final.dropna(subset=['lat', 'lon']).drop_duplicates().reset_index(drop=True)

    buffer = io.BytesIO()
    df_final.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    client = crear_cliente_minio()
    minio_subir_memoria(client, subcarpeta, nombre_fichero, buffer)
    print(f"Subido a minIO: {nombre_fichero} \n")

def identificar_tipo_ocio(row: pd.Series) -> str:
    """ Clasifica sitios de ocio según tags de OSM. """

    if pd.notna(row.get('shop')): return 'comercio'
    if pd.notna(row.get('leisure')): return 'deporte'
    if pd.notna(row.get('amenity')):
        if row['amenity'] in ['cinema', 'theatre', 'arts_centre', 'library']: return 'cultural'
        return 'vida_nocturna'
    return 'otros'

def identificar_tipo_negativo(row: pd.Series) -> str:
    """ Clasifica sitios negativos según tags de OSM. """

    if pd.notna(row.get('landuse')): return row['landuse']
    if pd.notna(row.get('amenity')): return row['amenity']
    if pd.notna(row.get('power')): return 'power_station'
    return 'otros'

if __name__ == "__main__":
    tags_ocio = {'shop': ['mall', 'department_store'], 
                 'leisure': ['fitness_centre', 'sports_centre'], 
                 'amenity': ['cinema', 'bar', 'pub']}
    
    tags_neg = {'landuse': ['industrial', 'landfill'],
                 'amenity': ['prison', 'grave_yard'],
                 'power': ['substation']}
    
    procesar_y_subir_osm(tags_ocio, "indicadores_ocio_madrid.parquet", "datos_secundarios/ocio")
    procesar_y_subir_osm(tags_neg, "indicadores_negativos_madrid.parquet", "datos_secundarios/negativos")