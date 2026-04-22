import tempfile
import webbrowser

import geopandas as gpd
import matplotlib.pyplot as plt
import time
import folium
import matplotlib
import mapclassify
from utils.config import COMPONENTES_TRANSPORTE, TIPOS_REJILLAS,MINIO_REJILLAS_SUCIO,PATH_PRIMARIOS_LIMPIO,ARCHIVOS_COORDENADAS,MODOS,ARCHIVOS_VIVIENDAS
from utils.funciones_minio import bajar_mapa_minio, buscar_todos_los_archivos,crear_cliente_minio,bajar_minio, subir_mapa_minio
import pandas as pd
import h3
from minio import Minio
from shapely.geometry import Polygon

diccionarios_columnas_rejillas = {
    "barrios":["NOMBRE","COD_BAR","AREA"],
    "secciones censales":['CUSEC', 'COD_BAR', 'AREA'],
    "hexagonos_1":['id_hex','AREA'],
    "hexagonos_2":['id_hex','AREA']
}


def descarga_rejilla(tipo:str,cliente:Minio)->gpd.GeoDataFrame:
    """
        Descarga una rejilla de raw del tipo introducido
    Args:
        tipo (str): tipo de rejilla (barrio,seccion_censal,hex1,hex2)
        cliente (Minio): Cliente de Minio

    Returns:
        gpd.GeoDataFrame: geodataframe de la rejilla del tipo
    """
    gdf = bajar_mapa_minio(cliente,MINIO_REJILLAS_SUCIO,f"{tipo.replace(' ','_')}_madrid")
    return gdf

def subir_rejilla_llena(cliente:Minio,gdf:gpd.GeoDataFrame,nombre_rejilla:str,path="rejillas"):
    """
        Sube la rejilla al minio una vez procesada
    Args:
        cliente (Minio): Cliente minio
        gdf (gpd.GeoDataFrame): geodataframe de la rejilla
        nombre_rejilla (str): nombre de la rejilla
        path (str, optional): path de donde subir la rejilla agrupadas. Defaults to "rejillas".
    """
    subir_mapa_minio(cliente,gdf,path,nombre_rejilla)

def extraer_mapa_principal(df_puntos:pd.DataFrame, gdf_mapa_completo:gpd.GeoDataFrame, id_columna:str, lat_col='lat', lon_col='lon')->gpd.GeoDataFrame:
    """
    Toma los puntos existentes, mira qué polígonos del mapa oficial tocan, 
    y devuelve un GeoDataFrame limpio solo con esos polígonos. Solo para barrios y secciones
    Args:
        df_puntos (pd.DataFrame): coordenadas existentes
        gdf_mapa_completo (gpd.GeoDataFrame): Mapa de rejilla completo
        id_columna (str): id de la columna que identifica una rejilla
        lat_col (str, optional):  Defaults to 'lat'.
        lon_col (str, optional):  Defaults to 'lon'.

    Returns:
        gpd.GeoDataFrame: geoDataframe limpip con solamente las secciones de las que tenemos datos
    """
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
    """
        Obtiene de minio los puntos del mapa de madrid de los que tenemos datos
    Args:
        client (Minio): Cliente de minio

    Returns:
        pd.DataFrame: Dataframe de coordenadas que poseemos
    """
    path = PATH_PRIMARIOS_LIMPIO
    archivo = ARCHIVOS_COORDENADAS

    df_coordenadas = bajar_minio(client,path,archivo)
    return df_coordenadas


def mete_datos_secundarios(gdf:gpd.GeoDataFrame,cod_rejilla:str,df_datos_sec:pd.DataFrame,nombre_cat:str)->gpd.GeoDataFrame:
    """
        Metemos datos secundarios en la rejilla introducida.
        Calcula para cada tipo de datos secundarios el número de puntos que se encuentran en cada sección de la rejilla, y la densidad (ese número penalizado por el área)
    Args:
        gdf (gpd.GeoDataFrame): gdf de la rejilla
        cod_rejilla (str): columna de identificacion de las secciones de las rejillas
        df_datos_sec (pd.DataFrame): dataframe de un tipo de datos secundarios
        nombre_cat (str): el tipo de datos que metemos (comercios, tiendas de alimentación,parques...)

    Returns:
        gpd.GeoDataFrame: geodataframe con los datos extraídos
    """

    df_datos = df_datos_sec.copy()
    gdf_res = gdf.copy()

    gdf_datos = gpd.GeoDataFrame(df_datos,geometry=gpd.points_from_xy(df_datos["lon"],df_datos["lat"]),crs="EPSG:4326")

    gdf_datos = gdf_datos.to_crs(gdf_res.crs)

    gdf_conjunto = gpd.sjoin(gdf_datos,gdf_res[[cod_rejilla,"geometry"]],how = 'inner',predicate = 'within')

    conteo = gdf_conjunto.groupby(cod_rejilla).size().reset_index(name=f"Num_{nombre_cat}")

    gdf_res = gdf_res.merge(conteo,on=cod_rejilla,how='left')

    gdf_res[f"Densidad_{nombre_cat}"] = round(gdf_res[f"Num_{nombre_cat}"]/gdf_res["AREA"],2)

    gdf_res[f"Densidad_{nombre_cat}"] = gdf_res[f"Densidad_{nombre_cat}"].fillna(0)
    gdf_res[f"Num_{nombre_cat}"] = gdf_res[f"Num_{nombre_cat}"].fillna(0)

    return gdf_res

def rellenar_nulos_con_vecinos(gdf_original:gpd.GeoDataFrame, columnas_a_imputar:list)->gpd.GeoDataFrame:
    """
        Función de interpolación con vecinos cercanos
        Rellena los valores nulos de un GeoDataFrame calculando la media de los polígonos vecinos que sí tienen datos.
        Por ahora nos centramos sobre todo en el precio por metro cuadrático medio de cada sección
    Args:
        gdf_original (gpd.GeoDataFrame): geodataframe de la rejilla
        columnas_a_imputar (list): columna a rellenar 

    Returns:
        gpd.GeoDataFrame: Devuelve un geodataframe con los precios por metro cuadrático calculado para todas
    """
    gdf = gdf_original.copy()
    
    for col in columnas_a_imputar:
        mask_nulos = gdf[col].isna()
        
        if not mask_nulos.any():
            continue 
                    
        for indice, fila_vacia in gdf[mask_nulos].iterrows():
            
            mask_vecinos = (
                gdf.geometry.intersects(fila_vacia.geometry) & 
                (gdf.index != indice) & 
                gdf[col].notna()
            )
            
            vecinos_validos = gdf[mask_vecinos]
            
            if not vecinos_validos.empty:
                media_vecindario = vecinos_validos[col].mean()
                gdf.loc[indice, col] = media_vecindario
                
    return gdf

def generar_rejilla_h3(df:gpd.GeoDataFrame, lat_col='lat', lon_col='lon', resolucion=8, anillos=3)->gpd.GeoDataFrame:
    """
        Genera rejilla de h2 a partir del df de coordenadas en función de la resolución. 
        El sistema de anillos permite generar los hexágonos vecinos para que el mapa sea uniforme 
    Args:
        df (gpd.GeoDataFrame): Df de coordenadas de viviendas de las que tenemos información
        lat_col (str, optional): columna de latitud. Defaults to 'lat'.
        lon_col (str, optional): columna de longitud. Defaults to 'lon'.
        resolucion (int, optional): controla el tamaño de los hexágonos. Defaults to 8.
        anillos (int, optional): cantidad de hexágonos vecinos que se generan por defecto. Defaults to 3.

    Returns:
        gpd.GeoDataFrame: la rejilla de hexágonos h3 generada
    """
    df_temp = df.copy()

    def obtener_h3(fila):
        return h3.latlng_to_cell(fila[lat_col], fila[lon_col], resolucion)  

    df_temp['id_hex'] = df_temp.apply(obtener_h3, axis=1)
    df_agrupado = df_temp.groupby('id_hex').size().reset_index(name='total_puntos')

    hex_activos = df_agrupado['id_hex'].unique()
    hex_expandidos = set()


    for hex_id in hex_activos: 
        vecinos = h3.grid_disk(hex_id, anillos)
        hex_expandidos.update(vecinos)  

    df_malla_completa = pd.DataFrame({'id_hex': list(hex_expandidos)})
    df_final = pd.merge(df_malla_completa, df_agrupado, on='id_hex', how='left')

    df_final['total_puntos'] = df_final['total_puntos'].fillna(0)


    def crear_poligono(hex_id):
        vertices_hex = h3.cell_to_boundary(hex_id) 

        return Polygon([(lon, lat) for lat, lon in vertices_hex])

    df_final['geometry'] = df_final['id_hex'].apply(crear_poligono)

    gdf_res= gpd.GeoDataFrame(df_final, geometry='geometry', crs="EPSG:4326")
    gdf_res['AREA'] = gdf_res['id_hex'].apply(lambda x: h3.cell_area(x, unit='km^2'))

    return gdf_res

def mete_datos_transporte(gdf_rejilla, df_transporte, col_id_rejilla, tipo_transporte, col_lineas='lineas', col_lat='lat', col_lon='lon'):
    """
    Cruza espacialmente paradas de transporte con una rejilla.
    Calcula cuántas paradas caen en cada celda y cuántas LÍNEAS ÚNICAS son accesibles.
    """
    gdf_res = gdf_rejilla.copy()
    
    if tipo_transporte == "paradas":
        lineas = "bus"
    else:
        lineas = "metro"

    gdf_transporte = gpd.GeoDataFrame(
        df_transporte, 
        geometry=gpd.points_from_xy(df_transporte[col_lon], df_transporte[col_lat]),
        crs="EPSG:4326" 
    )

    if gdf_transporte.crs != gdf_res.crs:
        gdf_transporte = gdf_transporte.to_crs(gdf_res.crs)

    cruce = gpd.sjoin(gdf_transporte, gdf_res[[col_id_rejilla, 'geometry']], how='inner', predicate='within')

    if cruce.empty:
        gdf_res[f'Num_{tipo_transporte}'] = 0
        gdf_res[f'Num_lineas_{lineas}'] = 0
        return gdf_res

    def contar_lineas_unicas(series_lineas):
        lineas_unicas = set()
        for item in series_lineas:
            lineas_unicas.update(item)
        return len(lineas_unicas)

    agrupado = cruce.groupby(col_id_rejilla).agg(
        num_paradas=(col_id_rejilla, 'size'), 
        num_lineas=(col_lineas, contar_lineas_unicas) 
    ).reset_index()

    agrupado = agrupado.rename(columns={
        'num_paradas': f'Num_{tipo_transporte}',
        'num_lineas': f'Num_lineas_{lineas}'
    })

    gdf_res = gdf_res.merge(agrupado, on=col_id_rejilla, how='left')

    for col in [f'Num_{tipo_transporte}', f'Num_lineas_{lineas}']:
        gdf_res[col] = gdf_res[col].fillna(0).astype(int)

    return gdf_res


def mete_datos_catastro(gdf_rejilla:gpd.GeoDataFrame,gdf_edificios:gpd.GeoDataFrame,id_rejilla:str)->gpd.GeoDataFrame:
    """
        Introduce los datos del catastro a las rejillas de barrios y secciones censales
    Args:
        gdf_rejilla (gpd.GeoDataFrame): rejilla base de la división de madrid
        gdf_edificios (gpd.GeoDataFrame): datos de la construcción de los edificios
        id_rejilla (str): id que se usa para identificar las distintas divisiones

    Returns:
        gpd.GeoDataFrame: Rejilla completada con los datos
    """

    gdf_res = gdf_rejilla.copy()

    if gdf_edificios.crs != gdf_res.crs:
        gdf_edificios = gdf_edificios.to_crs(gdf_res.crs)
    
    gdf_edificios_puntos = gdf_edificios.copy()

    gdf_edificios_puntos['geometry'] = gdf_edificios_puntos.geometry.representative_point()

    cruce = gpd.sjoin(gdf_edificios_puntos, gdf_res[[id_rejilla, 'geometry']], how='inner', predicate='within')
    
    agrupado = cruce.groupby(id_rejilla).agg(
        Anio_construccion=("anio_construccion", 'mean')
    ).reset_index()

    gdf_res = gdf_res.merge(agrupado, on=id_rejilla, how='left')

    return gdf_res

def mete_datos_ine(gdf_rejilla:gpd.GeoDataFrame,df_ine:pd.DataFrame,id_rejilla:str)->gpd.GeoDataFrame:
    """
        Introduce datos de la renta media madrileña a la rejilla
    Args:
        gdf_rejilla (gpd.GeoDataFrame): rejilla de división de madrid
        df_ine (pd.DataFrame): df de datos
        id_rejilla (str): id que se usa para identificar las distintas divisiones

    Returns:
        gpd.GeoDataFrame: Rejilla completada con los datos de renta media
    """
    
    gdf_res = gdf_rejilla.copy()

    agrupado = df_ine.groupby(id_rejilla).agg(
        Renta_media=("renta_media", 'mean')
    ).reset_index()

    agrupado['Renta_media'] = agrupado['Renta_media'].round(2)

    gdf_res = gdf_res.merge(agrupado, on=id_rejilla, how='left')

    return gdf_res   

def mete_datos_viviendas(gdf:gpd.GeoDataFrame,cod_rejilla:str,df_viviendas:pd.DataFrame,tipo:str)->gpd.GeoDataFrame:
    """
    Toma una rejilla (barrios,seccion_censal, h3) y un df de viviendas.
    Asigna una categoria a cada vivienda por Lat/Lon.
    Luego calcula medias, proporciones y devuelve el mapa listo.
    """
    minimo_pisos = 7
    df_viv = df_viviendas.copy()
    gdf_res = gdf.copy()

    gdf_viv = gpd.GeoDataFrame(df_viv,geometry=gpd.points_from_xy(df_viv["lon"],df_viv["lat"],crs = "EPSG:4326"))
    gdf_viv = gdf_viv.to_crs(gdf_res.crs)

    gdf_conjunto = gpd.sjoin(gdf_viv,gdf_res[[cod_rejilla,"geometry"]],how = 'inner',predicate='within')

    df_viv = pd.DataFrame(gdf_conjunto).drop(columns = ['geometry','index_right'])

    df_viv['Precio_m2'] = df_viv['Precio'] / df_viv['Superficie']
    if tipo == "alquiler":
        cols_boleanos = ['Ascensor','Terraza','Balcon','Equipamiento','Cocina']
    else:
        cols_boleanos = ['Ascensor','Terraza','Balcon']

    for col in cols_boleanos:
        if col in df_viv.columns:
            df_viv[col] = df_viv[col].astype(float)

    agrupaciones = {
        'Precio':'mean',
        'Superficie':'mean',
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
    df_agrupado[f'Media_precio_m2_{tipo}'] = df_agrupado[f'Media_precio_{tipo}'] / df_agrupado[f'Media_superficie_{tipo}']

    gdf_res = gdf_res.merge(df_agrupado, on=cod_rejilla, how='left')

    gdf_res[f'Num_viviendas_{tipo}'] = gdf_res[f'Num_viviendas_{tipo}'].fillna(0)
    gdf_res[f'Densidad_viviendas_{tipo}'] = round(gdf_res[f'Num_viviendas_{tipo}'] / gdf_res['AREA'], 2)

    mask_pocos_datos = gdf_res[f'Num_viviendas_{tipo}'] < minimo_pisos
    cols_a_limpiar = [f'Media_precio_m2_{tipo}']
    gdf_res.loc[mask_pocos_datos, cols_a_limpiar] = None
    gdf_res = rellenar_nulos_con_vecinos(gdf_res,cols_a_limpiar)
    for c in nuevos_nombres.values():
        gdf_res[c] = gdf_res[c].fillna(0)

    return gdf_res

def limpiar_coordenadas_lejanas(df, lat_min=40.35, lat_max=40.5, lon_min=-3.76, lon_max=-3.615):
    """
    Filtra los puntos que caen fuera de una 'caja' lógica.
    (Las coordenadas por defecto son un recuadro relativamente amplio alrededor de Madrid).
    """
    total_antes = len(df)
    
    df_limpio = df[
        (df['lat'] >= lat_min) & (df['lat'] <= lat_max) & 
        (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
    ].copy()
        
    return df_limpio

def descargar_viviendas(cliente:Minio,modo:str)->pd.DataFrame:
    df = bajar_minio(cliente,f"{PATH_PRIMARIOS_LIMPIO}/{modo}",ARCHIVOS_VIVIENDAS)
    return df

def descargar_datos(cliente:Minio,path:str,nombre_archivo:str)->pd.DataFrame:
    df = bajar_minio(cliente,path,nombre_archivo)
    return df


def calcula_area(gdf:gpd.GeoDataFrame):
    """
        Calcula el área en kilómetros cuadrados de las rejillas
    Args:
        gdf (gpd.GeoDataFrame): Geodataframe con el área de las divisiones calculadas
    """
    crs_activo = gdf.crs
    if crs_activo is None or crs_activo.to_epsg() == 4326:
        gdf = gdf.to_crs("EPSG:25830") 
        
    gdf['AREA'] = round(gdf["AREA"] / 1000000,5)
    
    if crs_activo is not None:
        gdf = gdf.to_crs(crs_activo)

def visualizar_rejilla(gdf:gpd.GeoDataFrame):
    """
        Función auxiliar que crea un html con el Mapa del geaodataframe que recibe para visualizarlo
    Args:
        gdf (gpd.GeoDataFrame): Mapa a visualizar
    """
    mapa_base = gdf.explore(column="Anio_construccion")
    folium.LayerControl().add_to(mapa_base)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        mapa_base.save(tmp.name)
        ruta_temporal = tmp.name
    webbrowser.open('file://' + ruta_temporal)

def inicio_rejillas():
    """
        Función que  ejecuta la totalidad de agrupación necesaria en el mapa de rejillas
    """
    cliente = crear_cliente_minio()
    df_coorenadas = obtener_coordenadas_procesadas(cliente)
    df_coorenadas = limpiar_coordenadas_lejanas(df_coorenadas)
    df_viviendas_venta = descargar_datos(cliente,PATH_PRIMARIOS_LIMPIO,"viviendas_venta.parquet")
    df_viviendas_alquiler = descargar_datos(cliente,PATH_PRIMARIOS_LIMPIO,"viviendas_alquiler.parquet")
    diccionario_transporte = {}
    for transporte in COMPONENTES_TRANSPORTE:
        df = descargar_datos(cliente,"cleaned/transporte",transporte["fichero"])
        diccionario_transporte[transporte["calculo"]] = df
    diccionario_viviendas = {
        "venta": df_viviendas_venta,
        "alquiler": df_viviendas_alquiler
    }
    df_ine = bajar_minio(cliente,"cleaned/ine","renta_media.parquet")
    gdf_padron_barrios = bajar_minio(cliente,"cleaned/padron","padron_barrio_madrid.parquet")
    gdf_padron_secciones = bajar_minio(cliente,"cleaned/padron","padron_seccion_madrid.parquet")
    gdf_catastro = bajar_mapa_minio(cliente,"cleaned/catastro","anio_construccion")
    datos_secundarios = buscar_todos_los_archivos(cliente,"cleaned/secundarios")
    diccionario_secundarios = {}
    for sec in datos_secundarios:
        df = descargar_datos(cliente,"cleaned/secundarios",sec)
        diccionario_secundarios[sec.removesuffix(".parquet")] = df
    for rejilla in TIPOS_REJILLAS:
        if "hexagonos" in rejilla["tipo"]:
            if "1" in rejilla["tipo"]:
                resolucion = 8
            else:
                resolucion = 9
            gdf_rejilla = generar_rejilla_h3(df_coorenadas,resolucion=resolucion)

        else:
            gdf_rejilla = descarga_rejilla(rejilla["tipo"],cliente)
            if rejilla["tipo"] == "barrios":
                gdf_rejilla = gdf_rejilla.merge(gdf_padron_barrios,on = rejilla["columna_id"],how = "left")
            else:
                gdf_rejilla = gdf_rejilla.merge(gdf_padron_secciones,on = rejilla["columna_id"],how = "left")
            gdf_rejilla = mete_datos_catastro(gdf_rejilla,gdf_catastro,rejilla["columna_id"])
            gdf_rejilla = mete_datos_ine(gdf_rejilla,df_ine,rejilla["columna_id"])
        calcula_area(gdf_rejilla)
        for modo in MODOS:
            gdf_rejilla = mete_datos_viviendas(gdf_rejilla,rejilla['columna_id'],diccionario_viviendas[modo],modo)
        for tipo,df_s in diccionario_secundarios.items():
            gdf_rejilla = mete_datos_secundarios(gdf_rejilla,rejilla["columna_id"],df_s,tipo)
        for tipo,df_t in diccionario_transporte.items():
            gdf_rejilla = mete_datos_transporte(gdf_rejilla,df_t,rejilla["columna_id"],tipo)
        subir_rejilla_llena(cliente,gdf_rejilla,rejilla["tipo"])
        print(f"Mapa de {rejilla["tipo"]} subido con exito.")

if __name__ == "__main__":
    inicio_rejillas()