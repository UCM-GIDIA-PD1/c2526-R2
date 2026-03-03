import osmnx as ox
import pandas as pd
import io
from src.utils.funciones_minio import crear_cliente_minio, minio_subir_memoria
from src.config import (
    MINIO_PROCESSED_SECUNDARIOS, PLACE_OSM, TAGS_COMERCIO, TAGS_NEGATIVOS, TAGS_ALIMENTACION,
    OBJ_COMERCIO, OBJ_NEGATIVOS, OBJ_ALIMENTACION,
)

'''
Script para la extracción de puntos de interés desde OpenStreetMap (OSM),
enfocado en indicadores de comercios en general (peluquerias, restaurantes...), negativos (industrias, vertederos, prisiones)
y alimentacion (supermercados, panaderias...).
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
    ox.settings.use_cache = True
    
    # Extracción
    features = ox.features.features_from_place(PLACE_OSM, tags=tags)
    
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
    if 'comercio' in nombre_fichero.lower():
        features['tipo'] = 'comercio'
    elif 'alimentacion' in nombre_fichero.lower():
        features['tipo'] = 'alimentacion'
    else:
        features['tipo'] = 'negativo'

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

if __name__ == "__main__":
    procesar_y_subir_osm(TAGS_COMERCIO, OBJ_COMERCIO, MINIO_PROCESSED_SECUNDARIOS)
    procesar_y_subir_osm(TAGS_NEGATIVOS, OBJ_NEGATIVOS, MINIO_PROCESSED_SECUNDARIOS)
    procesar_y_subir_osm(TAGS_ALIMENTACION, OBJ_ALIMENTACION,MINIO_PROCESSED_SECUNDARIOS)
