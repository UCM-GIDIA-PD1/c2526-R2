import osmnx as ox
import pandas as pd
import html
import re

# En este codigo usamos la libreria osmnx para sacar sitios de ocio y posibles lugares que puedan afectar negativamente al precio como cementerios o subestaciones electricas
ox.settings.use_cache = True

place = "Madrid, Spain"

tags_ocio = {
    # Comercios grandes
    'shop': ['mall', 'department_store'],

    # Deporte
    'leisure': ['fitness_centre', 'sports_centre', 'stadium', 'swimming_pool'],

    # Cultura
    'amenity': ['cinema', 'theatre', 'arts_centre', 'library'],

    # Vida Nocturna
    'amenity': ['nightclub', 'bar', 'pub']
}

ocio = ox.features.features_from_place(place, tags=tags_ocio)

# Calculamos el centroide (Para los sitios grandes q tengan mas de una coord)
ocio['lat'] = ocio.geometry.centroid.y
ocio['lon'] = ocio.geometry.centroid.x

columnas_utiles = ['name', 'lat', 'lon', 'amenity', 'shop']
ocio = ocio[[c for c in columnas_utiles if c in ocio.columns]]

ocio = ocio.fillna('')

print(ocio.head())
ocio.to_csv("puntos_ocio_madrid.csv", index=False)

ox.settings.use_cache = True

tags_negativos = {
    # Industria
    'landuse': ['industrial', 'landfill'],

    # Edificios específicos
    'amenity': ['prison', 'grave_yard', 'crematorium'],

    # Infraestructuras ruidosas o feas visualmente
    'power': ['substation', 'plant']
}

negativos = ox.features.features_from_place(place, tags=tags_negativos)

negativos['lat'] = negativos.geometry.centroid.y
negativos['lon'] = negativos.geometry.centroid.x

# Nos quedamos con el nombre y el tipo de "cosa negativa"
def identificar_tipo(row):
    if pd.notna(row.get('landuse')): return row['landuse']
    if pd.notna(row.get('amenity')): return row['amenity']
    if pd.notna(row.get('power')): return 'power_station'
    if pd.notna(row.get('railway')): return 'railway'
    return 'otros'

negativos['tipo_negativo'] = negativos.apply(identificar_tipo, axis=1)

cols_utiles = ['name', 'lat', 'lon', 'tipo_negativo']
df_final = negativos[[c for c in cols_utiles if c in negativos.columns]].copy()

# Rellenamos nombres vacíos con el tipo
df_final['name'] = df_final['name'].fillna(df_final['tipo_negativo'])

print(df_final.head())
df_final.to_csv("indicadores_negativos_madrid.csv", index=False)