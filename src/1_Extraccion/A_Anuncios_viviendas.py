import pandas as pd
from DrissionPage import ChromiumPage,WebPage
from DrissionPage.errors import ElementNotFoundError
from src.utils.funciones_minio import *
from src.config import URL_ALQUILER, URL_VENTA, MINIO_RAW_PRIMARIOS, IDEALISTA_UMBRAL_ANUNCIOS
import requests
from PIL import Image
import re
import io
import numpy as np
import time
import random
import os
from tqdm import tqdm
from pathlib import Path
from minio.error import S3Error 

umbral = IDEALISTA_UMBRAL_ANUNCIOS


def extraer_datos_anuncio(page:ChromiumPage,url:str)->dict:
    """_summary_
        Extrae los datos de la vivienda que corresponde a la pagina cargada en el anuncio
    Args:
        page (ChromiumPage): acceso a idealista con pagina cargada
        url (str): url de la vivienda

    Returns:
        dict: informacion de vivienda
    """

    vivienda = {
        'id':'',
        'Nombre':'',
        'Barrio':'',
        'Distrito':'',
        'Calle':'',
        'Precio':'',
        'Superficie':'',
        'Num_habitaciones':'',
        'Banyos':'',
        'Planta':'',
        'Ventanas':'',
        'Ascensor':'',
        'Terraza':'',
        'Balcon':'',
        'Equipamiento':'',
        'Cocina':'',
        'Orientacion':'',
        'Consumo':'',
        'Descripcion':'',
        'Anuncia':'',
        'Url':url,
        'Imagenes':''
    }
    page = corrige_page(page,url)
    info_primaria = page.ele('tag:section@class=detail-content-wrapper')
    if info_primaria:
        match = re.search(r'/inmueble/(\d+)/', url)
        vivienda['id'] = match.group(1) 
        vivienda['Nombre'] = info_primaria.ele('tag:span@class=main-info__title-main').text
        precio_loc = info_primaria.ele('tag:span@class=info-data-price')
        vivienda['Precio'] = precio_loc.ele('tag:span@class=txt-bold').text
        caract_box = page.ele('tag:section@class=details-box')
        page.scroll.to_see(caract_box)
        vivienda['Consumo'] = extraer_certificado_ennergetico(caract_box)
        vivienda['Descripcion'] = extraer_descripcion(info_primaria.ele('tag:div@class=comment'))
        vivienda['Anuncia'] = extrae_anunciante(page.ele('tag:div@id=module-contact-container'))
        vivienda['Imagenes'] = extraer_imagenes(page,page.ele('tag:div@id=main-multimedia'))
        for key,elem in extraer_caracteristicas(caract_box).items():
            if key in vivienda:
                vivienda[key] = elem
        for key,elem in extraer_datos_geograficos(page).items():
            if key in vivienda:
                vivienda[key] = elem
        return vivienda
    else:
        raise ElementNotFoundError(f"Contenedor primario no hallado en: {url}")

def controla_dataframe(df:pd.DataFrame)->pd.DataFrame:
    """

    Fuerza el tipado de las columnas que pueden quedar con anomalías

    Args:
        df (pd.DataFrame): El dataframe con un conjutno de viviendas

    Returns:
        pd.DataFrame: Dataframe con todas las columnas tipadas
    """
    columnas_numericas = ['Num_habitaciones','Banyos','Planta']
    df_res = df.copy()
    for col in columnas_numericas:
        if col in df_res.columns:
            df_res[col] = pd.to_numeric(df_res[col],errors='coerce')
    return df_res

def extraer_certificado_ennergetico(info)->str:
    """_summary_
        Extrae el certificado de consumo energetico del anuncio
    Args:
        info (ElementoChromiumPage): contenedor en el que encuentran las caracteristicas de la vivienda

    Returns:
        str: consumo de la vivienda (Letra de la etiqueta correspondiente)
    """
    certificado = info.ele('tag:div@class=energy-certificate-dropdown')
    if certificado :
        contenedor = info.ele('tag:div@class=details-property-feature-two')
        features = contenedor.eles('tag:h2@class=details-property-h2')
        infos = contenedor.eles('tag:div@class=details-property_features')
        for cat,seccion in zip(features,infos):
            if cat.text == 'Certificado energético':
                consumo = seccion.ele('tag:span@class').attr('class').removeprefix('icon-energy-c-').upper()
                return consumo
        
    else:
        return 'No determinado'
    

def extraer_imagenes(page:ChromiumPage,contenedor)->list:
    """_summary_
        Extrae las imagenes controlando el numero de imagenes por habitacon, solicitandolas y pasandolas a formato bytes
    Args:
        page (ChromiumPage): acceso a idealista con pagina cargada
        contenedor (ElementoChromiumPage): contenedor en el que se encuentran las imagenes

    Returns:
        list: lista de diccionario de imagenes con el nombre de a que habitacion corresponde
    """
    page.scroll.to_see(contenedor)

    imagenes = contenedor.eles('css:picture')
    titulos = contenedor.eles('tag:span@class=detail-gallery-tag-name --center')
    cont = {
        'Dormitorio':0,
        'Cocina':0,
        'Salón':0,
        'Comedor':0,
        'Baño':0
    }
    lista = []
    total = 0
    session = requests.Session()
    for imagen,titulo in zip(imagenes,titulos):
        if titulo.text in cont and cont[titulo.text] < 2 and total < 8:
            src = imagen.ele('tag:source@type=image/webp').attr('srcset')
            request = session.get(src)
            if request.status_code == 200:
                img_data = io.BytesIO(request.content)
                with Image.open(img_data) as im:
                    if im.mode in ("RGBA", "P"):
                        im = im.convert("RGB")
                    buffer_webp = io.BytesIO()
                    im.save(buffer_webp, format="WebP", quality=80, method=6)
                    lista.append({titulo.text:buffer_webp.getvalue()})
                    total+=1
        elif total >= 8:
            break
    return lista

def ver_anuncio(vivienda:dict):
    """_summary_
        Visualizar la vivienda con imagenes aparte 
    Args:
        vivienda (dict): vivienda con su informacion
    """
    for key,value in vivienda.items():
        if key != "Imagenes":
            print(f"{key}:{value}")

def extrae_anunciante(info)->str:
    """
        Extrae el tipo de anunciante del anuncio de una vivienda (Profesional o Particular )
    Args:
        info (ElementoChromiumPage): contenedor que contiene la información del anunciante

    Returns:
        str: Nombre del tipo de anunciante
    """
    contenedor = info.ele('tag:div@class=ide-box-contact module-contact-gray contact-data-container ')
    nombre = contenedor.ele('tag:div@class=professional-name').ele('tag:div@class=name')
    if nombre.text:
        return nombre.text
    else:
        return "Particular" 

def extraer_caracteristicas(info)->dict:
    """
    Extrae las características básicas de una vivienda que se encuentran en el bloque características
    en idealista

    Args:
        info (ElementoChromiumPage): contenedor de las características

    Returns:
        dict: características básicas de una vivienda
    """
    caracteristicas = {
        'Superficie':None,
        'Num_habitaciones':None,
        'Banyos':None,
        'Planta':None,
        'Ventanas':None,
        'Ascensor':None,
        'Terraza':None,
        'Balcon':None,
        'Equipamiento':None,
        'Cocina':None,
        'Orientacion':None
    }

    enteros = ['Num_habitaciones','Banyos','Planta']
    no_determinados = ["Orientacion",'Ventanas']
    categoria = info.ele('tag:div@class=details-property-feature-one')
    if categoria:
        lista = categoria.eles('tag:li')
        for cat in lista:
            res = caracteristica(cat.text)
            for key,elem in res.items():
                if key in caracteristicas and caracteristicas[key] is None:
                    caracteristicas[key] = elem
    
    for k,v in caracteristicas.items():
        if k in enteros and caracteristicas[k] is None:
            caracteristicas[k] = 0
    for k,v in caracteristicas.items():
        if k in no_determinados and caracteristicas[k] is None:
            caracteristicas[k] = "no determinado"
    caracteristicas = {k:(False if v is None and k != "Orientacion" else v) for k,v in caracteristicas.items()}

    return caracteristicas

def extraer_datos_geograficos(info)->dict:
    """
        Extrae los datos geográficos que proporciona idealista de la vivienda 
        Calle - Barrio - Distrito
    Args:
        info (ElementoChromiumPage): contenedor con los datos geográficos

    Returns:
        dict: diccionario con los datos geográficos extraídos
    """
    ubicacion = {
        'Barrio':None,
        'Distrito':None,
        'Calle':None
    }

    contenedor = info.ele('tag:div@id=headerMap')
    ubicacion['Calle'] = contenedor.ele('tag:li').text
    for cat in contenedor.eles('tag:li'):
        if 'Barrio' in cat.text and ubicacion['Barrio'] is None:
            ubicacion['Barrio']= " ".join(cat.text.split()[1:])
        elif 'Distrito' in cat.text:
            ubicacion['Distrito'] =  " ".join(cat.text.split()[1:])

    return ubicacion


def extraer_descripcion(contenedor)->str:
    """
        Extrae la descripción de la vivienda

    Args:
        contenedor (ElementoChromiumPage): Contenedor con la descripción

    Returns:
        str: string de descripción sin saltos de línea
    """
    seccion_descripcion = contenedor.ele('tag:p')
    return seccion_descripcion.text.replace('\n',' ')

def caracteristica(texto)->dict:
    """
        Extrae de los textos de cracaterísticas la que corresponde al texto
    Args:
        texto (str): texto que es una línea del bloque de las características

    Returns:
        dict: diccionario con tantas entradas como características se han extraído
    """
    if 'm²' in texto:
        numero = re.findall(r'-?\d+\.?\d*',texto)
        numero = max(numero)
        return {'Superficie':numero}
    elif 'habitaciones' in texto or 'habitación' in texto:
        numero = re.findall(r'-?\d+\.?\d*',texto)
        if len(numero) == 0:
            numero = 0
        else:
            numero = max(numero)
        return {'Num_habitaciones':numero}
    elif 'baños' in texto or 'baño' in texto:
        numero = re.findall(r'-?\d+\.?\d*',texto)
        if len(numero) == 0:
            numero = 0
        return {'Banyos':max(numero)}
    elif 'Amueblado' in texto or 'cocina equipada' in texto or 'Cocina equipada' in texto:
        res = {
            'Equipamiento':None,
            'Cocina':None
        }
        res['Cocina'] = 'cocina equipada' in texto or 'Cocina equipada' in texto
        res['Equipamiento'] = 'Amueblado' in texto
        return res
    elif 'Balcón' in texto or 'balcón' in texto or 'Terraza' in texto or 'terraza' in texto:
        res={   
            'Terraza':None,
            'Balcon':None
            }
        res['Balcon'] = 'Balcón' in texto or 'balcón' in texto
        res['Terraza']='Terraza' in texto or 'terraza' in texto
        return res
    elif 'Planta' in texto or 'exterior' in texto or 'interior' in texto or "Exterior" in texto or "Interior" in texto:
        res={   
            'Planta':None,
            'Ventanas':None
            }
        planta = re.findall(r'-?\d+\.?\d*',texto)
        if len(planta) == 0:
            res['Planta'] = "bajo"
        else:
            res['Planta'] = max(planta)
        ventana = texto.split()[-1:]
        if "exterior" in texto or "interior" in texto or "Exterior" in texto or "Interior" in texto :
            res['Ventanas'] = ventana[0]
        else:
            res['Ventanas'] = "no indicado"
        return res
    elif 'ascensor' in texto:
        ascensor = 'Con' in texto
        return {'Ascensor':ascensor}
    elif 'Orientación' in texto:
        orientacion = texto.split()[1:]
        return {'Orientacion':orientacion[0]}
    else:
        return {'info_inutil':None}
        

def corrige_page(page:ChromiumPage,url:str)->ChromiumPage:
    """
        Función para evitar lo errores de generación de los html y también en caso de detección por parte de idealista del script
        Vuelve a ejecutar el html hasta que se genere bien autmáticamente o que lo corrija un usuario pasando el test de verificación
    Args:
        page (ChromiumPage): el acceso a la página web
        url (str): la url que estamos tratando de obtener
    Returns:
        ChromiumPage: Devuelve el acceso a la página de la url 
    """
    pagina_valida = page.ele('tag:div@id=wrapper')
    while not pagina_valida:
        time.sleep(random.uniform(5, 10))
        page.get(url)
        pagina_valida = page.ele('tag:body').attr('class')
    return page



def links_regiones(page:ChromiumPage,url:str,regiones_unicos:set,num:0)->list:
    """
        Recursiva de obtención de los links de regiones y subregiones
        trata de ampliar las regiones mayor que la umbral de 1200 anuncios y hacer que no haya repeticion de zonas 
    Args:
        page (ChromiumPage): acceso a la pagina web
        url (str): url de la pagina activa en el momento
        regiones_unicos (set): set de las distintas regiones
        num (0): numero de anuncios por zona

    Returns:
        list: lista de diccionario con {"region","link","num"} donde num es el numero de anuncios dentro de dicha zona
    """
    page = corrige_page(page,url)
    if page.ele('tag:main@class=listing-items  core-vitals-listing-map'):
        return {"link":page.url.astype(str),"num":num}
    else:
        expandir = page.ele('.sublocations-showall')
        zonas_expandir = []
        páginas_zonas = []
        if expandir:
            boton_mostrar_todo = expandir.ele('tag:a')
            page.scroll.to_see(boton_mostrar_todo)
            boton_mostrar_todo.click()
            time.sleep(random.uniform(0.5, 2))
            contenedor = page.ele('.navList nav-list')
        else:
            contenedor = page.ele('.navList')
        lista_zonas = contenedor.eles('tag:li')
        barra_progreso = tqdm(lista_zonas,desc="Extrayendo links de regiones")
        for zona in barra_progreso :
            if zona.ele('tag:a') :
                link = zona.ele('tag:a').attr('href').removesuffix('mapa')
                if "alquiler" in link :
                    region = link.removeprefix("https://www.idealista.com/alquiler-viviendas/madrid/")[:-1].replace('/',' - ')
                else:
                    region = link.removeprefix("https://www.idealista.com/venta-viviendas/madrid/")[:-1].replace('/',' - ')
                if region.split(' - ')[0] not in regiones_unicos and region not in regiones_unicos:
                    if int(zona.ele("tag:span@class=subdued").text.replace(".","")) < umbral :
                        páginas_zonas.append({"region":region,"link":link,"num":int(zona.ele("tag:span@class=subdued").text.replace(".",""))})
                        regiones_unicos.add(region)
                    else:
                        zonas_expandir.append({"region":region,"link":zona.ele('tag:a').attr('href'),"num":int(zona.ele("tag:span@class=subdued").text.replace(".",""))})
        for urls in zonas_expandir:
            if urls not in páginas_zonas:
                print(f"Extrayendo subregiones de la region madrid - {urls["region"]}")
                page.get(urls["link"])
                páginas_zonas.extend(links_regiones(page,urls["link"],regiones_unicos,urls["num"]))

        return páginas_zonas


def sacar_link(anuncio,lista:set)->dict:
    """
        Extrae el link del anuncio que se le pasa
    Args:
        anuncio (ElementoChromiumPage): contenedor del anuncio 
        lista (set): set de los links de anuncios ya extraídos oo existentes en el minio para evitar repeticiones

    Returns:
        dict: diccionario con {"nombre":,"anuncio":,"id":}
    """
    tag_a = anuncio.ele('tag:a@role=heading')
    if tag_a:
        link = tag_a.attr('href')
        if link is not None:
            match = re.search(r'/inmueble/(\d+)/', link)
            id = match.group(1)
            if id not in lista:
                nombre = tag_a.attr('title')
                return {"nombre":nombre, "anuncio":link,"id":id}
    return None

def analizar_pagina(page:ChromiumPage,lista_ids:set)->list:
    """
        Extrae todos los links de anuncios de una página html de idealista.
        Además añade los ids a una lista para evitar que causen repetición
    Args:
        page (ChromiumPage): acceso a la pagina
        lista_ids (set): set de ids para no repetir anuncios

    Returns:
        list: lista de diccionarios con {"nombre":,"anuncio":,"id":} correspondientes a una pagina de anuncios
    """
    pagina = []
    time.sleep(random.uniform(1, 3))
    anuncios = page.eles('tag:article')
    for anuncio in anuncios:
        clase = anuncio.attr('class')
        if 'adv' in clase:
            continue
        anuncio_nuevo = sacar_link(anuncio,lista_ids)
        if anuncio_nuevo:
            pagina.append(anuncio_nuevo)
            lista_ids.add(anuncio_nuevo["id"])
    return pagina

def guardar_pagina_en_csv(lista_diccionarios:list, ruta_archivo:str)->None:
    """
        [AUXILIAR] : Para guardar en memoria las paginas de anuncios y visualizar
    Args:
        lista_diccionarios (list): lista de anuncios con sus links
        ruta_archivo (str): ruta de archivo
    """
    if not lista_diccionarios:
        return
    
    df = pd.DataFrame(lista_diccionarios)
    
    df = df[['nombre', 'anuncio']]
    
    archivo_existe = os.path.isfile(ruta_archivo)
    
    df.to_csv(ruta_archivo, mode='a', index=False, header=not archivo_existe, encoding='utf-8')

def guardas_links_regiones(lista_zonas:list,ruta:str)->None: 
    """
    [AUXILIAR] : Para guardar en memoria la lista de regiones y visualizar

    Args:
        lista_zonas (list): lista de regiones con sus links
        ruta (str): ruta de archivo
    """
    df = pd.DataFrame(lista_zonas)
        
    df = df[['link', 'num']]
        
    archivo_existe = os.path.isfile(ruta)
        
    df.to_csv(ruta, mode='a', index=False, header=not archivo_existe, encoding='utf-8')

def guardas_viviendas(lista_viviendas:list)->None:
    """
        [AUXILIAR] : Para guardar en memoria la lista de viviendas y visualizar
    Args:
        lista_viviendas (list): lista de viviendas con todas sus características
    """
    home = Path.home()

    ruta_test = home / "test_maiday.csv"
    df = pd.DataFrame(lista_viviendas)
    df = controla_dataframe(df)
    archivo_existe = os.path.isfile(ruta_test)
        
    df.to_csv(ruta_test, mode='a', index=False, header=not archivo_existe, encoding='utf-8')

def obtiene_anuncios(links_regiones:dict,page:ChromiumPage,anuncios_unicos:set)->dict:
    """
        Extrae todos los anuncios de las regiones que se encutran en links_regiones
    Args:
        links_regiones (dict): {"region","link"} para la region que se extrae
        page (ChromiumPage): acceso a chrome para acceder las regiones
        anuncios_unicos (set): conjunto de ids ya tratados hasta el momento

    Returns:
        dict: diccionario con entrada de {"region":[{},{},{}]} donde los diccionarios internos son los distintos anuncios encontrados
    """
    res = {}
    for url in links_regiones:
        siguiente = True
        page.get(url["link"])
        page = corrige_page(page,url["link"])
        res[url["region"]] = []
        barra_progreso = tqdm(desc=f"Extrayendo anuncios de la region de {url["region"]}")
        while siguiente:
            next = page.ele('.next')
            res[url["region"]].extend(analizar_pagina(page,anuncios_unicos))
            if not next:
                siguiente = False
            else:
                url_next = next.ele('tag:a').attr('href')
                barra_progreso.update(1)
                page.get(url_next)
        barra_progreso.close()
    return res

def imprimir_header()->None:
    """
    Header inicial del scrapper
    """
    print("╔" + "═" * 58 + "╗")
    print("║" + "MAiDay SCRAPER".center(58) + "║")
    print("║" + "v11.1".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

def imprimir_menu_modo()->None:
    """
        Menu para seleccion de modo inicial
    """
    print("\n" + " Selecciona el tipo de mercado ".center(60, "░"))
    print("\n   [A]  VENTA")
    print("   [B]  ALQUILER")
    print("   [C]  ACTUALIZACION IDs ANTES DE SCRAPPEAR")
    print("\n" + "─" * 60)

def imprimir_regiones(lista_datos:list)->list:
    """
        Imprime  las regiones disponibles para scrapear y devuelve la o las seleccionadas
    Args:
        lista_datos (list): Lista de las regiones de madrid con sus links y numero de viviendas

    Returns:
        list: lista de regiones con sus links y numeros de viviendas
    """
    total_anuncios = sum(item['num'] for item in lista_datos)
    
    print("\n" + "  MAPA DE EXTRACCIÓN ".center(56, "▒"))
    print(f"{'ID':<4} │ {'REGIÓN':<25} │ {'ANUNCIOS':>12}")
    print("─" * 56)

    for i, data in enumerate(lista_datos, 1):
        nombre = data['region']
        cantidad = f"{data['num']:,}".replace(",", ".")
        
        print(f"{i:<4} │ {nombre:<40} │ {cantidad:>10} ")

    print("─" * 56)
    
    total_str = f"{total_anuncios:,}".replace(",", ".")
    
    print("\n          ╔════════════════════════════════════════╗")
    print(f"          ║   [0]    PROCESAR TODAS LAS ZONAS     ║")
    print(f"          ║        Acumulado: {total_str:>12} ads     ║")
    print("          ╚════════════════════════════════════════╝\n")
    print(f"          ║   [-1]    PARA SALIR AL MENU PRINCIPAL     ║")
    while True:
        entrada = input("\n Introduce el ID (o '0' para todas): ").strip()

        if not entrada.replace('-','').isdigit():
            print(" Error: Por favor, introduce un número, no letras.")
            continue

        opcion = int(entrada)
        if opcion == 0:
            print(f" Has seleccionado: [TODAS LAS REGIONES]")
            return lista_datos

        if opcion == -1:
            print("\n Volviendo al inicio...")
            return []

        if 1 <= opcion <= len(lista_datos):
            seleccionada = lista_datos[opcion - 1]
            nombre = seleccionada['region']
            
            print(f" Has seleccionado: [{nombre}]")
            return [seleccionada]

        print(f" El ID {opcion} no existe en la lista. Prueba otra vez.")

def obtener_siguiente_indice(client: Minio, modo:str, region:str, umbral_mb=340):
    """
        Obtiene el parquet sobre el que se va a operar. 
        O decide recupar uno ya existente y descargarlo
        O no existe uno que corresponde a la misma region y crea uno nuevo
        O encuentra los de la misma region y los descarta por ya ser muy grandes > umbral-mb(default 340 MB)
    Args:
        client (Minio client): cliente de minio 
        modo (str): tipo de mercado sobre el que estoy operando (venta o alquiler)
        region (str): nombre de region sobre la que estoy operando
        umbral_mb (int, optional): Umbral de que tan grande tiene que ser un parquet para que decida no recuperarlo Defaults to 340.

    Returns:
        int: numero de archivo sobre el que se va a operar
        bool: si tenemos o no que descragar el archivo
    """
    bucket = os.getenv("MINIO_BUCKET")
    group_path = os.getenv("MINIO_GROUP_PATH")
    prefix = f"{group_path}/raw/datos_primarios/{modo}/"
    umbral_bytes = umbral_mb * 1024 * 1024
    max_indice = 0
    size_ultimo = 0
    
    objetos = client.list_objects(bucket, prefix=prefix, recursive=True)
    
    for obj in objetos:
        if region in obj.object_name and obj.object_name.endswith(".parquet"):
            match = re.search(r'_n_(\d+)', obj.object_name)
            if match:
                indice_actual = int(match.group(1))
                if indice_actual > max_indice:
                    max_indice = indice_actual
                    size_ultimo = obj.size 

    if max_indice == 0:
        return 1,False 
        
    if size_ultimo < umbral_bytes:
        return max_indice,True
    else:
        return max_indice + 1,False

def obtener_ids_existentes(client: Minio, modo: str) -> set:
    """
        Obtiene todos los ids existentes de viviendas de un tipo de mercado (alquiler o venta) y los descarga.
        Este es el método bruto pasando por todos los parquets.
        A ejecutar una vez cada semana para garantizar que no haya repetición
    Args:
        client (Minio): cliente de minio
        modo (str): tipo de mercado sobre el que estoy operando (venta o alquiler)

    Returns:
        set: conjunto de ids que ya existen en la nube
    """
    bucket = os.getenv("MINIO_BUCKET")
    group_path = os.getenv("MINIO_GROUP_PATH")
    prefix = f"{group_path}/raw/datos_primarios/{modo}/"
    
    ids_totales = set()
    objetos = client.list_objects(bucket, prefix=prefix, recursive=True)
    progreso = tqdm(objetos,desc = "Extrayendo ids de anuncios ya descargados")
    for obj in progreso:
        if obj.object_name.endswith(".parquet"):
            response = client.get_object(bucket, obj.object_name)
            try:
                buffer = io.BytesIO(response.read())
                df_temp = pd.read_parquet(buffer, columns=['id'])
                
                ids_totales.update(df_temp['id'].astype(str).tolist())
            except Exception as e:
                print(f" Error leyendo {obj.object_name}: {e}")
            finally:
                response.close()
                response.release_conn()
    print(f"     Extraídos {len(ids_totales)} anuncios para evitar repetición    ")
  
    return ids_totales

def analiza_lista(lista_anuncios:list,region:str,ids_unicos:set,page:ChromiumPage,cliente:Minio,modo:str):
    """
        Analiza todas las viviendas en una lista de viviendas correspondientes a una region de un tipo de mercado
        Realiza el avance descargando cada vez que el dataframe sea muy grande
        Garantiza que los anuncios de errores se ignores pero que se sigan manteniendo en la lista de anuncios por si se quieren volver a tratar
    Args:
        lista_anuncios (list): lista de anuncios
        region (str): region sobre la que se etsá operando
        ids_unicos (set): conjunto de anuncios unicos
        page (ChromiumPage): Aceeso a la pagina de chrome
        cliente (Minio): cliente de minio
        modo (str): tipo de mercado (venta o alquiler)

    Returns:
        int,int,int: numero de archivos que se han subido, numero de anuncios que se han extraido, numero de errores que se han producido
    """
    errores = 0
    lista_viviendas = []
    df_total = pd.DataFrame()
    cont_arch,descargar = obtener_siguiente_indice(cliente,modo,region.replace(' - ', '_'))
    if descargar:
        path,nombre_fichero = construye_path(modo,region,cont_arch)
        df_total = bajar_minio(cliente,path,nombre_fichero)
        print(f"             .parquet de {region} recuperado          ")
    cont_anuncios = 0
    progreso = tqdm(lista_anuncios,desc=f"Analizando anuncios de la zona {region}...")
    lista_errores = []
    for vivienda in progreso:
        nombre = vivienda["nombre"]
        url = vivienda["anuncio"]
        try:
            page.get(url)
            lista_viviendas.append(extraer_datos_anuncio(page,url))
            cont_anuncios+=1
            if len(lista_viviendas) >= 30:
                df = pd.DataFrame(lista_viviendas)
                df_total = pd.concat([df_total, df], ignore_index=True)
                lista_viviendas = []
            if len(df_total) >= 500 :
                df_total = controla_dataframe(df_total)
                buffer = io.BytesIO()
                df_total.to_parquet(buffer, engine="pyarrow", index=False)
                buffer.seek(0)
                path,nombre_fichero = construye_path(modo,region,cont_arch)
                subir_viviendas(modo,buffer,region,ids_unicos,cont_arch,cliente)
                print(f"             Se ha subido el avance en {nombre_fichero}.parquet         ")
                df_total = pd.DataFrame()
                cont_arch+=1
        except ElementNotFoundError:
            # Error específico: La página cargo pero el dato no esta (piso borrado, diseño distinto o muy frecuentemente sin imágenes)
            errores += 1
            progreso.set_postfix(Error = errores,Solucion = "Salto de anuncio") 
            ids_unicos.remove(vivienda["id"])
            lista_errores.append(vivienda)
            continue 
        except Exception as e:
            # Error genérico: Se cerró el navegador, se fue el internet, etc.
            errores += 1
            print(f"\n Error crítico en {nombre},{url}: {e}")
            ids_unicos.remove(vivienda["id"])
            lista_errores.append(vivienda)
            continue
    if (not df_total.empty or lista_viviendas) and cont_anuncios:
        df = pd.DataFrame(lista_viviendas)
        df_total = pd.concat([df_total, df], ignore_index=True)
        buffer = io.BytesIO()
        df_total = controla_dataframe(df_total)
        df_total.to_parquet(buffer, engine="pyarrow", index=False)
        buffer.seek(0)
        subir_viviendas(modo,buffer,region,ids_unicos,cont_arch,cliente)
        path,nombre_fichero = construye_path(modo,region,cont_arch)
        print(f"             Se ha subido el final en {nombre_fichero}.parquet         ")
    lista_anuncios[:] = lista_errores
    return cont_arch,cont_anuncios,errores

def construye_path(modo:str,region:str,batch:int)->str:
    """
        construye el path crrespondiente a nuestro parquet tanto para descragar como subir
    Args:
        modo (str): _description_
        region (str): _description_
        batch (int): _description_

    Returns:
        str: _description_
    """
    if modo == "venta":
        path = MINIO_RAW_PRIMARIOS +"/"+  "venta"
    elif modo == "alquiler":
        path = MINIO_RAW_PRIMARIOS +"/"+ "alquiler"
    nombre_region = region.replace(' - ', '_')
    nombre_fichero = f"batch_{nombre_region}_n_{batch}.parquet"

    return path,nombre_fichero

def subir_viviendas(modo:str,buffer: io.BytesIO,region:str,ids_unicos:set,batch:int,cliente:Minio)-> None:
    """
        Almacena el batch de viviendas recibido al minio
        Actualiza los ids de anuncios ya extraidos
    Args:
        modo (str): tipo de mercado (alquiler o venta)
        buffer (io.BytesIO): buffer conteniendo los anuncios
        region (str): region de madrid a la que corresponden las vivientas
        ids_unicos (set): ids de anuncios extraidos
        batch (int): numero de archivos almacenados para esa zona
        cliente (Minio): cliente de minio
    """
    path,nombre_fichero = construye_path(modo,region,batch)
    actualizar_ids(cliente,modo,ids_unicos)
    minio_subir_memoria(cliente,path,nombre_fichero,buffer)

def descargar_ids(client:Minio, modo: str) -> set:
    """
       Descarga el Parquet maestro de IDs desde MinIO y lo devuelve como un set de Python.
       Si el archivo no existe (primera ejecución), devuelve un set vacío.
    Args:
        client (Minio): cliente de minio
        modo (str): tipo de mercado (alquiler o venta)

    Raises:
        e: error de no encontrar la lista de ids

    Returns:
        set: conjunto de ids extraídos hasta el momento
    """
    bucket = os.getenv("MINIO_BUCKET")
    group_path = os.getenv("MINIO_GROUP_PATH")
    
    object_name = f"{group_path}/raw/datos_primarios/{modo}/ids.parquet"
    
    try:
        response = client.get_object(bucket, object_name)
        buffer = io.BytesIO(response.read())
        
        df_ids = pd.read_parquet(buffer, columns=['id'])
        
        ids_set = set(df_ids['id'].astype(str).tolist())
        
        print(f" Recuperados {len(ids_set)} IDs de {modo}.")
        return ids_set
        
    except S3Error as e:
        if e.code == 'NoSuchKey':
            print(f" No existe archivo maestro para '{modo}'. Se creará uno nuevo.")
            return set()
        else:
            print(f" Error crítico de MinIO: {e}")
            raise e
            
    finally:
        if 'response' in locals():
            response.close()
            response.release_conn()

def escanear_y_corregir_duplicados(client, modo: str) -> set:
    bucket = os.getenv("MINIO_BUCKET")
    group_path = os.getenv("MINIO_GROUP_PATH")
    prefix = f"{group_path}/raw/datos_primarios/{modo}/"
    
    mapa_ids = {} # Diccionario para rastrear { 'id_anuncio': 'nombre_del_fichero.parquet' }
    archivos_a_limpiar = set() # Aquí guardamos los ficheros que tienen ids repetidas
    duplicados_encontrados = 0
    
    objetos = client.list_objects(bucket, prefix=prefix, recursive=True)
    
    for obj in tqdm(objetos,desc="Recuperando los parquets existentes en la nube"):
        nombre_archivo = obj.object_name
        
        if not nombre_archivo.endswith(".parquet") or "ids" in nombre_archivo:
            continue
            
        response = client.get_object(bucket, nombre_archivo)
        try:
            buffer = io.BytesIO(response.read())
            df_temp = pd.read_parquet(buffer, columns=['id'])
            lista_ids = df_temp['id'].astype(str).tolist()
            
            # Buscamos duplicados en este archivo
            for idx in lista_ids:
                if idx in mapa_ids:
                    duplicados_encontrados += 1
                    archivos_a_limpiar.add(nombre_archivo)
                else:
                    mapa_ids[idx] = nombre_archivo
                    
        finally:
            response.close()
            response.release_conn()

    print(f" Escaneo completado. IDs únicos reales: {len(mapa_ids)}")
    
    if archivos_a_limpiar:
        print(f"  Se han detectado {duplicados_encontrados} duplicados.")
        
        for archivo_sucio in archivos_a_limpiar:
            limpiar_archivo_minio(client, bucket, archivo_sucio)
            
        print(" Limpieza terminada. Los duplicados han sido eliminados de la nube.")
    else:
        print(" Lus datos están inmaculados. 0 duplicados.")

    actualizar_ids(client,modo,set(mapa_ids.keys()))

    return set(mapa_ids.keys())


def limpiar_archivo_minio(client, object_name,modo:str):
    """Descarga un parquet, le quita los duplicados y lo vuelve a subir."""
    print(f"   ->  Limpiando .parquet: {object_name}")
    
    path = f"raw/datos_primarios/{modo}"
    df_sucio = bajar_minio(client,path,object_name)
    filas_antes = len(df_sucio)
    
    df_limpio = df_sucio.drop_duplicates(subset=['id_anuncio'], keep='first')
    filas_despues = len(df_limpio)
    
    if filas_antes > filas_despues:
        print(f"      Borradas {filas_antes - filas_despues} filas duplicadas.")
        
        buffer_out = io.BytesIO()
        df_limpio.to_parquet(buffer_out, engine='pyarrow', index=False)
        buffer_out.seek(0)
        
        minio_subir_memoria(client,path,object_name,buffer_out)


def actualizar_ids(client:Minio, modo: str, ids_set: set) -> None:
    """
            Convierte un set de IDs a Parquet y sobrescribe el archivo  en MinIO.

    Args:
        client (Minio): cliente de minio
        modo (str): tipo de mercado (venta o alquiler)
        ids_set (set): set de ids extraídos hasta el momento
    """
    print(f" Guardando {len(ids_set)} IDs en el archivo de {modo}...")
    
    df_ids = pd.DataFrame({'id': list(ids_set)})
    
    df_ids['id'] = df_ids['id'].astype(str)
    
    buffer = io.BytesIO()
    df_ids.to_parquet(buffer, engine='pyarrow', index=False)
    buffer.seek(0)
    path = f"raw/datos_primarios/{modo}"
    minio_object = "ids.parquet"
    
    try:
        minio_subir_memoria(client, path, minio_object, buffer)
    except Exception as e:
        print(f"Falló la actualización del archivo de ids: {e}")
    finally:
        buffer.close()

def inicio():
    """
        menu inicial para selección de mercado sobre el que operamos
    Returns:
        str: tipo de mercado seleccionado
        str: url correspondiente al tipo de mercado seleccionado
        set: set descargado desde minio de los ids ya extraídos
        Minio: cliente de minio 
    """
    imprimir_header()
    imprimir_menu_modo()
    cliente = crear_cliente_minio()
    
    modo_input = input("Selecciona un modo: ")
    entradas = ['A', 'a', 'B', 'b', 'C', 'c']
    
    while modo_input not in entradas:
        print(" Error de entrada")
        imprimir_menu_modo()
        modo_input = input("Selecciona un modo: ")
        
    if modo_input in ['A', 'a']:
        url = URL_VENTA
        modo = "venta"
        anuncios_unicos = descargar_ids(cliente, modo)
        
    elif modo_input in ['B', 'b']:
        url = URL_ALQUILER
        modo = "alquiler"
        anuncios_unicos = descargar_ids(cliente, modo)
        
    elif modo_input in ['C', 'c']:
        print(" Actualizacion de ides")
        print("\n" + " Selecciona el tipo de mercado ".center(60, "░"))
        print("\n   [A]  VENTA")
        print("   [B]  ALQUILER")
        modo_input = input("Selecciona un modo: ").upper()
        
        while modo_input not in ['A', 'a','b','B']:
            print(" Error de entrada")
            imprimir_menu_modo()
            modo_input = input("Selecciona un modo: ")
            
        if modo_input in ['A', 'a']:
            url = URL_VENTA
            modo = "venta"
        
        elif modo_input in ['B', 'b']:
            url = URL_ALQUILER
            modo = "alquiler"

        anuncios_unicos = escanear_y_corregir_duplicados(cliente, modo)
        
        actualizar_ids(cliente, modo, anuncios_unicos)
        
        print(f" Memoria reconstruida con {len(anuncios_unicos)} IDs. Lista para scrapear.")

    return modo, url, anuncios_unicos,cliente

def webscraping_idealista()-> None:
    """
        Ejecución del webscraping
        Función a llamar para ejecutar la extracción
    """

    modo,url,anuncios_unicos,cliente = inicio()
    
    page = ChromiumPage()
    page.get(url)
    regiones_unicas = set()
    links_regiones_madrid = links_regiones(page,url,regiones_unicas,0)
    regiones_interesadas = imprimir_regiones(links_regiones_madrid)
    if len(regiones_interesadas) == links_regiones_madrid:
        for region in regiones_interesadas:
            lista_anuncios = obtiene_anuncios(region,page,anuncios_unicos)
            for clave,lista in lista_anuncios.items():
                archivos,anuncios,errores = analiza_lista(lista,clave,anuncios_unicos,page,cliente,modo) 
                if archivos == -1:
                    archivos = 1
                print(f"    Se han subido {archivos} archivos de {anuncios} anuncios y descartando {errores} anuncios con errores   ") 
                pos = next((i for i , d in enumerate(links_regiones_madrid) if d["region"] == regiones_interesadas[0]["region"]),None)
                if not lista:
                    links_regiones_madrid.pop(pos)
                else:
                    links_regiones_madrid[pos]["num"] = len(lista)
    else:
        while len(regiones_interesadas) > 0:
            lista_anuncios = obtiene_anuncios(regiones_interesadas,page,anuncios_unicos)
            for clave,lista in lista_anuncios.items():
                archivos,anuncios,errores = analiza_lista(lista,clave,anuncios_unicos,page,cliente,modo) 
                if archivos == -1:
                    archivos = 1
                print(f"    Se han subido {archivos} archivos de {anuncios} anuncios y descartando {errores} anuncios con errores   ")
                pos = next((i for i , d in enumerate(links_regiones_madrid) if d["region"] == regiones_interesadas[0]["region"]),None)
                if not lista:
                    links_regiones_madrid.pop(pos)
                else:
                    links_regiones_madrid[pos]["num"] = len(lista)
            regiones_interesadas = imprimir_regiones(links_regiones_madrid)


if __name__ == '__main__':
    webscraping_idealista()