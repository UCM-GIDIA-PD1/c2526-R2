import googlemaps
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from utils.config import PAIS,CIUDAD
import re
from utils.funciones_minio import *
from tqdm import tqdm
import os
from utils.config import COLUMNAS_CARACTERISTICAS,COLUMNAS_IMAGENES,PATH_PRIMARIOS_LIMPIO,PATH_PRIMARIOS_RAW,ARCHIVOS_COORDENADAS,ARCHIVOS_VIVIENDAS,ARCHIVOS_IMAGENES,MODOS

COLUMNAS_SELECCION_DUPLICADOS = ["Superficie","Precio","Num_habitaciones","Ventanas","Planta","Anuncia","Direccion","lat","lon"]

def limpia_direccion(direccion:str)->str:
    """
        limpia la cadena de "Calle" de una vivienda y la normaliza para que pueda pasar a coordenadas
    Args:
        direccion (str): Calle donde se encuentra

    Returns:
        str: Direccion de la vivienda con ciudad y Pais
    """
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
    """
        Descarga los anuncios correspondientes a un tipo de mercado de la carpeta raw
    Args:
        client (Minio): Cliente de minio
        modo (str): tipo de mercado que se quiere descargar

    Returns:
        pd.DataFrame: dataframe de las viviendas con solo las columnas interesantes para el modelo (todo menos imágenes)
    """
    path = f'{PATH_PRIMARIOS_RAW}{modo}'
    parquets = buscar_todos_los_archivos(client,path)
    ignorar = {'ids.parquet'}
    df_res = pd.DataFrame()
    for parquet in tqdm(parquets,desc = "Descargando totalidad de anuncios"):
        if parquet not in ignorar:
            df = bajar_minio_especifico(client,path,parquet.replace(path,''),columnas=COLUMNAS_CARACTERISTICAS)
            df_res = pd.concat([df_res,df],ignore_index=True)
    
    return df_res

def subir_viviendas_limpio(df:pd.DataFrame,cliente:Minio,modo:str):
    """
        Sube viviendas de un tipo de mercado (modo) en limpio al minio tras haber transformado y limpiado los datos
    Args:
        df (pd.DataFrame): df de viviendas
        cliente (Minio): cliente minio
        modo (str): Tipo de mercado (venta o alquiler)
    """
    path = PATH_PRIMARIOS_LIMPIO
    archivo = f"{ARCHIVOS_VIVIENDAS}_{modo}.parquet"   
    subir_minio(df,cliente,path,archivo)

def sustituir_valores_nulos(df:pd.DataFrame)->pd.DataFrame:
    """
        Hace la limpieza del datasets de viviendas :
        - Paso a nmérico de todo lo que lo sea
        - Quitar outliers (por ahora un solo parámetro (superficie > 10000m2))
    Args:
        df (pd.DataFrame): df de viviendas

    Returns:
        pd.DataFrame: df de viviendas limpio
    """

    df_limpiado = df.copy()

    df_limpiado['Planta'] = df_limpiado['Planta'].fillna(0)

    df_limpiado["Precio"] = df_limpiado["Precio"].astype(str).str.replace('.','')
    df_limpiado["Precio"] = pd.to_numeric(df_limpiado["Precio"],errors='coerce')
    df_limpiado["Superficie"] = df_limpiado["Superficie"].astype(str).str.replace('.','')
    df_limpiado["Superficie"] = pd.to_numeric(df_limpiado["Superficie"],errors='coerce')
    df_limpiado["Orientacion"] = (df_limpiado["Orientacion"].astype(str).str.replace(',','',regex = False).str.strip().str.capitalize())

    columnas_nulas = ['ventanas', 'orientacion']

    for col in columnas_nulas:
        if col in df_limpiado.columns:
            df_limpiado[col] = df_limpiado[col].astype(str).str.replace(r'(?i)(no determinado|no indicado)','No determinado', regex=True)

            df_limpiado[col] = df_limpiado[col].replace('nan', 'No determinado')
            df_limpiado[col] = df_limpiado[col].replace('Nan', 'No determinado')

    cantidad_duplicados = df.duplicated(subset = COLUMNAS_SELECCION_DUPLICADOS).sum()

    print(f"Detectando y borrando {cantidad_duplicados} anuncios muy similares ")
    df_limpiado = df_limpiado[df_limpiado["Superficie"]<=10000].copy()
    df_res = df_limpiado.drop_duplicates(subset = COLUMNAS_SELECCION_DUPLICADOS,keep = 'first').copy()

    return df_res

def calcular_precio_m2(df:pd.DataFrame)->None:
    """
        Calcula el precio por metro cuadrático de cada vivienda
    Args:
        df (pd.DataFrame): df de viviendas
    """
    df["Precio_m2"] = round(df["Precio"] / df["Superficie"], 5)

def obtener_coordenadas_procesadas(client:Minio)->pd.DataFrame:
    """
        Obtiene las coordenadas de viviendas que ya se han procesado 
        Hacemos esto para optimizar el proces y no Geocodificar cada vez que ejecutemos la limpieza
    Args:
        client (Minio): Cliente de minio

    Returns:
        pd.DataFrame: df de coordenadas
    """
    path = PATH_PRIMARIOS_LIMPIO
    archivo = ARCHIVOS_COORDENADAS
    try:
        print(f" Buscando memoria de coordenadas en: {path}/{archivo}")
        df_coordenadas = bajar_minio(client,path,archivo)
        print(f" Cargadas {len(df_coordenadas)} calles conocidas.")
        return df_coordenadas
    except Exception as e:
        print(f" Error al procesar la memoria: {e}")
        return pd.DataFrame()

def subir_coordenadas(client:Minio,df_coordenadas:pd.DataFrame)->None:
    """
        Sube las coordenadas de las que se ha encontrado la posición geográfica
    Args:
        client (Minio): Cliente de minio
        df_coordenadas (pd.DataFrame): coordenadas en un dataframe
    """
    path = PATH_PRIMARIOS_LIMPIO
    archivo = ARCHIVOS_COORDENADAS
    subir_minio(df_coordenadas,client,path,archivo)

def descargar_imagenes(cliente:Minio,path:str,nombre_archivo:str)->pd.DataFrame:
    """
        Descrgar las imágenes (columas : [id,Imagenes])
    Args:
        cliente (Minio): cliente de minio
        path (str): path de donde buscar el archivo
        nombre_archivo (str): nombre del archivo del que queremos descargar las imágenes

    Returns:
        pd.DataFrame: df de imágenes
    """
    df = bajar_minio_especifico(cliente,path,nombre_archivo,COLUMNAS_IMAGENES)
    df = aplanar_columnas_imagenes(df)
    return df

def aplanar_columnas_imagenes(df:pd.DataFrame):
    """
    Transforma la columna de diccionarios de imágenes en 5 columnas independientes.
    Cada columna contendrá una lista de bytes (las imágenes de esa habitación).
    Args:
        df (pd.DataFrame): df de imágenes

    """    
    nuevas_filas = []
    for datos in df["Imagenes"]:
        fila_habitaciones = {
            'Dormitorio': [],
            'Cocina': [],
            'Salón': [],
            'Comedor': [],
            'Banyo': []
        }
        
        for diccionario in datos:
            for habitacion, bytes_img in diccionario.items():
                nombre_columna = habitacion
                if habitacion == "Baño":
                    nombre_columna = "Banyo"
                if bytes_img is not None and nombre_columna in fila_habitaciones and isinstance(bytes_img, bytes):
                    fila_habitaciones[nombre_columna].append(bytes_img)
                        
        nuevas_filas.append(fila_habitaciones)
        
    df_expandido = pd.DataFrame(nuevas_filas, index=df.index)
    
    df_final = pd.concat([df.drop(columns=["Imagenes"]), df_expandido], axis=1)
    
    return df_final

def separar_imagenes(cliente:Minio):
    """
        Recorre la totalidad de ficheros raw que tenemos de las viviendas y las descarga invocando a aplanar_columnas_imagenes
        Sube al minio en la carpeta que correpsonde batchs de imagenes con tamaño controlado
    Args:
        cliente (Minio): Cliente de minio
    """
    for modo in MODOS:
        num_archivo = 1
        path_sucio = f"{PATH_PRIMARIOS_RAW}{modo}"
        parquets = buscar_todos_los_archivos(cliente,path_sucio)
        df_buffer = pd.DataFrame()
        for parquet in tqdm(parquets,desc=f"Transfiriendo imagenes de {modo} a limpio..."):
            if parquet != "ids.parquet":
                df_buffer = pd.concat([df_buffer,descargar_imagenes(cliente,path_sucio,parquet)])
                if len(df_buffer)>500:
                    df_subir = df_buffer.iloc[:500].copy()
                    subir_minio(df_subir,cliente,f"{PATH_PRIMARIOS_LIMPIO}/imagenes/{modo}",f"{ARCHIVOS_IMAGENES}_n_{num_archivo}.parquet")
                    num_archivo+=1
                    df_buffer = df_buffer.iloc[500:].reset_index(drop=True)
            


def aportar_coordenadas(df_venta:pd.DataFrame,df_alquiler:pd.DataFrame,cliente:Minio):
    """
        Toma un DataFrame, limpia la columna de direcciones y añade columnas de Latitud, Longitud y Tipo.
    Args:
        df_venta (pd.DataFrame): dataframe de viviendas de venta
        df_alquiler (pd.DataFrame): dataframe de viviendas de alquiler
        cliente (Minio): Cliente de minio

    Returns:
        pd.Dataframe: coordenadas procesadas con columnas [Direccion,lat,lon,tipo_osm,tipo_Vía]
    """

    tqdm.pandas(desc=" Geocodificando pisos")

    geolocator = Nominatim(user_agent="maiday_bot_v1")
    
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)
    load_dotenv()
    api_key = os.getenv("API_GOOGLE")
    gmaps = googlemaps.Client(key=api_key)

    def buscar_en_google(direccion):
        direccion_limpia = limpia_direccion(direccion)
        tipo_espanol = direccion_limpia.split(' ')[0]
        try:
                
            resultado = gmaps.geocode(direccion_limpia)
            
            if resultado: # Si Google encuentra algo
                lat = resultado[0]['geometry']['location']['lat']
                lon = resultado[0]['geometry']['location']['lng']
                tipos_lista = resultado[0].get('types', [])
                tipo_str = ", ".join(tipos_lista) if tipos_lista else "Desconocido"
                
                return pd.Series([lat, lon,tipo_str,tipo_espanol])
        except Exception as e:
            pass # Si falla por red o límite de API, devolvemos nulo
            
        return pd.Series([ None, None,"No Encontrado",tipo_espanol])
    
    def procesar_fila_osm(direccion_sucia):
        dir_limpia = limpia_direccion(direccion_sucia)
        tipo_espanol = dir_limpia.split(' ')[0]
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
                
                return pd.Series([lat, lon, tipo_nominatim, tipo_espanol])
            else:
                return pd.Series([None, None, "No Encontrado", tipo_espanol])
                    
        except Exception as e:
            return pd.Series([None, None, "Error API", tipo_espanol])

    df_alquiler['Direccion'] = df_alquiler['Calle'].apply(lambda x: limpia_direccion(x))
    df_venta['Direccion'] = df_venta['Calle'].apply(lambda x: limpia_direccion(x))
    df_coordenadas = obtener_coordenadas_procesadas(cliente)

    if not df_coordenadas.empty:
        mask_validas = df_coordenadas['lat'].notna() & (~df_coordenadas['Tipo_OSM'].isin(["No Encontrado", "Error API"]))
        df_coordenadas = df_coordenadas[mask_validas].copy()

    direcciones_alquiler = df_alquiler[["Direccion"]].dropna()
    direcciones_venta = df_venta[["Direccion"]].dropna()

    df_unicas = pd.concat([direcciones_alquiler,direcciones_venta]).drop_duplicates().reset_index(drop=True)

    if not df_coordenadas.empty:
        calles_conocidas = df_coordenadas["Direccion"].tolist()
        df_unicas = df_unicas[~df_unicas["Direccion"].isin(calles_conocidas)].copy()

    if not df_unicas.empty:
        print(f" Iniciando geocodificación de {len(df_unicas)} anuncios...")
        df_unicas[['lat', 'lon', 'Tipo_OSM', 'Tipo_Via']] = df_unicas["Direccion"].progress_apply(procesar_fila_osm)
        mask_fallos = df_unicas['lat'].isna() | df_unicas['Tipo_OSM'].isin(["No Encontrado", "Error API"])
        df_fallos = df_unicas[mask_fallos].copy()
        if not df_fallos.empty:
            df_fallos[['lat', 'lon', 'Tipo_OSM', 'Tipo_Via']] = df_fallos['Direccion'].progress_apply(buscar_en_google)
            df_unicas.update(df_fallos)
        mask_final_validas = df_unicas['lat'].notna() & (~df_unicas['Tipo_OSM'].isin(["No Encontrado", "Error API"]))
        df_unicas = df_unicas[mask_final_validas].copy()
    
    df_res = pd.concat([df_unicas,df_coordenadas],ignore_index=True)
    subir_coordenadas(cliente,df_res)
    print(" Geocodificación terminada.")

    return df_res

def limpiar_memoria_raw():
    """
        función principal de la limpieza de datos de viviendas. 
        Su ejecución deja en limpio y separadas viviendas_alquiler,viviendas_venta y carpeta imágenes(separando tipos de mercado)
    """
    cliente = crear_cliente_minio()
    df_alquiler = descargar_anuncios(cliente,"alquiler")
    df_venta = descargar_anuncios(cliente,"venta")
    df_coordenadas = aportar_coordenadas(df_venta,df_alquiler,cliente)
    df_venta = pd.merge(df_venta,df_coordenadas,on = "Direccion",how = 'left')
    df_alquiler = pd.merge(df_alquiler,df_coordenadas,on = "Direccion",how = 'left')
    df_venta = sustituir_valores_nulos(df_venta)
    df_alquiler = sustituir_valores_nulos(df_alquiler)
    calcular_precio_m2(df_venta)
    calcular_precio_m2(df_alquiler)
    subir_viviendas_limpio(df_alquiler,cliente,"alquiler")
    subir_viviendas_limpio(df_venta,cliente,"venta")
    separar_imagenes(cliente)
    print("Limpieza de datasets de viviendas completada.") 


if __name__=="__main__":
    limpiar_memoria_raw()