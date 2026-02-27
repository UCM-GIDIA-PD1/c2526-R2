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
    contenedor = info.ele('tag:div@class=ide-box-contact module-contact-gray contact-data-container ')
    nombre = contenedor.ele('tag:div@class=professional-name').ele('tag:div@class=name')
    if nombre.text:
        return nombre.text
    else:
        return "Particular" 

def extraer_caracteristicas(info):
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

def extraer_datos_geograficos(info):
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


def extraer_descripcion(contenedor):
    seccion_descripcion = contenedor.ele('tag:p')
    return seccion_descripcion.text.replace('\n',' ')

def caracteristica(texto):
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
        

def corrige_page(page,url):
    pagina_valida = page.ele('tag:div@id=wrapper')
    while not pagina_valida:
        time.sleep(random.uniform(5, 10))
        page.get(url)
        pagina_valida = page.ele('tag:body').attr('class')
    return page

umbral = IDEALISTA_UMBRAL_ANUNCIOS

def links_regiones(page,url,regiones_unicos,num:0):
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
            time.sleep(random.uniform(1, 3))
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


def sacar_link(anuncio,lista):
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

def analizar_pagina(page,lista_ids):
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

def guardar_pagina_en_csv(lista_diccionarios, ruta_archivo):
    if not lista_diccionarios:
        return
    
    df = pd.DataFrame(lista_diccionarios)
    
    df = df[['nombre', 'anuncio']]
    
    archivo_existe = os.path.isfile(ruta_archivo)
    
    df.to_csv(ruta_archivo, mode='a', index=False, header=not archivo_existe, encoding='utf-8')

def guardas_links_regiones(lista_zonas,ruta): 
       
    df = pd.DataFrame(lista_zonas)
        
    df = df[['link', 'num']]
        
    archivo_existe = os.path.isfile(ruta)
        
    df.to_csv(ruta, mode='a', index=False, header=not archivo_existe, encoding='utf-8')

def guardas_viviendas(lista_viviendas):
    home = Path.home()

# Creamos la ruta completa (puedes mandarlo al Escritorio para verlo rápido)
    ruta_test = home / "test_maiday.csv"
    df = pd.DataFrame(lista_viviendas)
        
    archivo_existe = os.path.isfile(ruta_test)
        
    df.to_csv(ruta_test, mode='a', index=False, header=not archivo_existe, encoding='utf-8')

def obtiene_anuncios(links_regiones,page,anuncios_unicos):
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

def imprimir_header():
    print("╔" + "═" * 58 + "╗")
    print("║" + "MAiDay SCRAPER".center(58) + "║")
    print("║" + "v1.0.2 - 2026 Edition".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

def imprimir_menu_modo():
    print("\n" + " Selecciona el tipo de mercado ".center(60, "░"))
    print("\n   [A]  VENTA")
    print("   [B]  ALQUILER")
    print("\n" + "─" * 60)

def imprimir_regiones(lista_datos):
    """
    lista_datos: [{'region': 'Madrid', 'num': 1500}, {'region': 'Chamberí', 'num': 250}]
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

        if not entrada.isdigit():
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

def obtener_siguiente_indice(client, modo, region, umbral_mb=350):
    """
    Busca el último índice de la región. 
    Si el último archivo es menor al umbral, devuelve ese mismo índice para sobreescribirlo/completarlo.
    Si es mayor o no existe, devuelve el siguiente.
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

def analiza_lista(lista_anuncios,region,page,cliente,modo):
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
            if df_total.memory_usage(deep=True).sum()/ (1024**2) >= 350:
                buffer = io.BytesIO()
                df_total = controla_dataframe(df_total)
                df_total.to_parquet(buffer, engine="pyarrow", index=False)
                buffer.seek(0)
                subir_viviendas(modo,buffer,region,cont_arch,cliente)
                df_total = pd.DataFrame()
                cont_arch+=1
        except ElementNotFoundError:
            # Error específico: La página cargo pero el dato no esta (piso borrado o diseño distinto)
            errores += 1
            progreso.set_postfix(Error = errores,Solucion = "Salto de anuncio") 
            continue 
        except Exception as e:
            # Error genérico: Se cerró el navegador, se fue el internet, etc.
            errores += 1
            print(f"\n Error crítico en {nombre},{url}: {e}")
            continue
    if not df_total.empty or lista_viviendas:
        df = pd.DataFrame(lista_viviendas)
        df_total = pd.concat([df_total, df], ignore_index=True)
        buffer = io.BytesIO()
        df_total = controla_dataframe(df_total)
        df_total.to_parquet(buffer, engine="pyarrow", index=False)
        buffer.seek(0)
        subir_viviendas(modo,buffer,region,cont_arch,cliente)
    return cont_arch,cont_anuncios,errores

def construye_path(modo:str,region:str,batch:int)->str:
    if modo == "venta":
        path = MINIO_RAW_PRIMARIOS +"/"+  "venta"
    elif modo == "alquiler":
        path = MINIO_RAW_PRIMARIOS +"/"+ "alquiler"
    nombre_region = region.replace(' - ', '_')
    nombre_fichero = f"batch_{nombre_region}_n_{batch}.parquet"

    return path,nombre_fichero

def subir_viviendas(modo:str,buffer: io.BytesIO,region:str,batch:int,cliente:Minio):
    
    path,nombre_fichero = construye_path(modo,region,batch)

    minio_subir_memoria(cliente,path,nombre_fichero,buffer)



def webscraping_idealista():
    imprimir_header()
    imprimir_menu_modo()
    modo = input()
    while modo != 'A' and modo !=  'a' and modo != 'B' and modo != 'b':
        print("Error de entrada")
        imprimir_menu_modo()
        modo = input()
    if modo == 'A' or modo == 'a':
        url = URL_VENTA
        modo = "venta"
    elif modo == 'B' or modo == 'b':
        url = URL_ALQUILER
        modo = "alquiler"
    
    page = ChromiumPage()
    page.get(url)
    cliente = crear_cliente_minio()
    regiones_unicas = set()
    links_regiones_madrid = links_regiones(page,url,regiones_unicas,0)
    anuncios_unicos = obtener_ids_existentes(cliente,modo)
    regiones_interesadas = imprimir_regiones(links_regiones_madrid)
    if len(regiones_interesadas) == links_regiones_madrid:
        for region in regiones_interesadas:
            lista_anuncios = obtiene_anuncios(region,page,anuncios_unicos)
            for clave,lista in lista_anuncios.items():
                archivos,anuncios,errores = analiza_lista(lista,clave,page,cliente,modo) 
                if archivos == -1:
                    archivos = 1
                print(f"    Se han subido {archivos} archivos de {anuncios} anuncios y descartando {errores} anuncios con errores   ") 
    else:
        while len(regiones_interesadas) > 0:
            lista_anuncios = obtiene_anuncios(regiones_interesadas,page,anuncios_unicos)
            for clave,lista in lista_anuncios.items():
                archivos,anuncios,errores = analiza_lista(lista,clave,page,cliente,modo) 
                if archivos == -1:
                    archivos = 1
                print(f"    Se han subido {archivos} archivos de {anuncios} anuncios y descartando {errores} anuncios con errores   ")
            pos = next((i for i , d in enumerate(links_regiones_madrid) if d["region"] == regiones_interesadas[0]["region"]),None)
            links_regiones_madrid.pop(pos)
            regiones_interesadas = imprimir_regiones(links_regiones_madrid)


if __name__ == '__main__':
    webscraping_idealista()