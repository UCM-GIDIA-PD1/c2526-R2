import tempfile
import webbrowser

import geopandas as gpd
import matplotlib.pyplot as plt
import time
import folium
import matplotlib
import mapclassify
from src.config import COMPONENTES_TRANSPORTE, TIPOS_REJILLAS,MINIO_REJILLAS_SUCIO,PATH_PRIMARIOS_LIMPIO,ARCHIVOS_COORDENADAS,MODOS,ARCHIVOS_VIVIENDAS
from src.utils.funciones_minio import bajar_mapa_minio, buscar_todos_los_archivos,crear_cliente_minio,bajar_minio, subir_mapa_minio
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


def descarga_rejilla(tipo:str,cliente:Minio):
    gdf = bajar_mapa_minio(cliente,MINIO_REJILLAS_SUCIO,f"{tipo.replace(' ','_')}_madrid")
    return gdf

def subir_rejilla_llena(cliente:Minio,gdf:gpd.GeoDataFrame,nombre_rejilla:str,path="rejillas"):
    subir_mapa_minio(cliente,gdf,path,nombre_rejilla)

def extraer_mapa_principal(df_puntos, gdf_mapa_completo, id_columna, lat_col='lat', lon_col='lon'):
    """
    Toma tus puntos, mira qué polígonos del mapa oficial tocan, 
    y te devuelve un GeoDataFrame limpio solo con esos polígonos.
    Ideal para Barrios y Secciones Censales.
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
    path = PATH_PRIMARIOS_LIMPIO
    archivo = ARCHIVOS_COORDENADAS

    df_coordenadas = bajar_minio(client,path,archivo)
    return df_coordenadas




def mete_datos_secundarios(gdf:gpd.GeoDataFrame,cod_rejilla:str,df_datos_sec:pd.DataFrame,nombre_cat:str)->gpd.GeoDataFrame:

    df_datos = df_datos_sec.copy()
    gdf_res = gdf.copy()

    gdf_datos = gpd.GeoDataFrame(df_datos,geometry=gpd.points_from_xy(df_datos["lon"],df_datos["lat"]),crs="EPSG:4326")

    gdf_datos = gdf_datos.to_crs(gdf_res.crs)

    gdf_conjunto = gpd.sjoin(gdf_datos,gdf_res[[cod_rejilla,"geometry"]],how = 'inner',predicate = 'within')

    conteo = gdf_conjunto.groupby(cod_rejilla).size().reset_index(name=f"Num_{nombre_cat}")

    gdf_res = gdf_res.merge(conteo,on=cod_rejilla,how='left')

    gdf_res[f"Densidad_{nombre_cat}"] = round(gdf_res[f"Num_{nombre_cat}"]/gdf_res["AREA"],2)


    return gdf_res

def rellenar_nulos_con_vecinos(gdf_original, columnas_a_imputar):
    """
    Rellena los valores nulos de un GeoDataFrame calculando la media 
    de los polígonos vecinos que sí tienen datos.
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

def generar_rejilla_h3(df, lat_col='lat', lon_col='lon', resolucion=8, anillos=3):
    """
    Genera una malla H3 continua alrededor de tus datos reales.
    Usa anillos de expansión para rellenar los huecos y poner los bordes a 0,
    evitando dibujar hexágonos inútiles a kilómetros de distancia.
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
    for c in gdf_res.columns:
        gdf_res[c] = gdf_res[c].fillna(0)

    return gdf_res

def limpiar_coordenadas_lejanas(df, lat_min=40.35, lat_max=40.5, lon_min=-3.76, lon_max=-3.615):
    """
    Filtra los puntos que caen fuera de una 'caja' lógica.
    (Las coordenadas por defecto son un recuadro amplio alrededor de Madrid).
    """
    total_antes = len(df)
    
    df_limpio = df[
        (df['lat'] >= lat_min) & (df['lat'] <= lat_max) & 
        (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
    ].copy()
    
    eliminados = total_antes - len(df_limpio)
    
    return df_limpio

def descargar_viviendas(cliente:Minio,modo:str)->pd.DataFrame:
    df = bajar_minio(cliente,f"{PATH_PRIMARIOS_LIMPIO}/{modo}",ARCHIVOS_VIVIENDAS)
    return df

def descargar_datos(cliente:Minio,path:str,nombre_archivo:str)->pd.DataFrame:
    df = bajar_minio(cliente,path,nombre_archivo)
    return df


def calcula_area(gdf:gpd.GeoDataFrame):
    crs_activo = gdf.crs
    if crs_activo is None or crs_activo.to_epsg() == 4326:
        gdf = gdf.to_crs("EPSG:25830") 
        
    gdf['AREA'] = round(gdf["AREA"] / 1000000,5)
    
    if crs_activo is not None:
        gdf = gdf.to_crs(crs_activo)

def mete_datos_mapa(gdf:gpd.GeoDataFrame,cliente:Minio)->gpd.GeoDataFrame:
    for sector in MODOS:
        df = descargar_viviendas(cliente,sector)

def visualizar_rejilla(gdf:gpd.GeoDataFrame,tipo:str):
    mapa_base = gdf.explore(column="Media_precio_m2_alquiler")
    folium.LayerControl().add_to(mapa_base)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        mapa_base.save(tmp.name)
        ruta_temporal = tmp.name
    webbrowser.open('file://' + ruta_temporal)

def inicio_rejillas():
    cliente = crear_cliente_minio()
    df_coorenadas = obtener_coordenadas_procesadas(cliente)
    df_coorenadas = limpiar_coordenadas_lejanas(df_coorenadas)
    df_viviendas_venta = descargar_datos(cliente,"datos_primarios","viviendas_venta.parquet")
    df_viviendas_alquiler = descargar_datos(cliente,"datos_primarios","viviendas_alquiler.parquet")
    diccionario_transporte = {}
    for transporte in COMPONENTES_TRANSPORTE:
        df = descargar_datos(cliente,"cleaned/transporte",transporte["fichero"])
        diccionario_transporte[transporte["calculo"]] = df
    diccionario_viviendas = {
        "venta": df_viviendas_venta,
        "alquiler": df_viviendas_alquiler
    }
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
            calcula_area(gdf_rejilla)
            for modo in MODOS:
                gdf_rejilla = mete_datos_viviendas(gdf_rejilla,rejilla['columna_id'],diccionario_viviendas[modo],modo)
            for tipo,df_s in diccionario_secundarios.items():
                gdf_rejilla = mete_datos_secundarios(gdf_rejilla,rejilla["columna_id"],df_s,tipo)
            for tipo,df_t in diccionario_transporte.items():
                gdf_rejilla = mete_datos_transporte(gdf_rejilla,df_t,rejilla["columna_id"],tipo)
        else:
            gdf_rejilla = descarga_rejilla(rejilla["tipo"],cliente)
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