import pandas as pd
from DrissionPage import ChromiumPage,WebPage
import requests
from PIL import Image
import re
import io
import numpy as np
import time
import random
import os
from tqdm import tqdm


url_alquiler = 'https://www.idealista.com/alquiler-viviendas/madrid-madrid/mapa'
url_venta = 'https://www.idealista.com/venta-viviendas/madrid-madrid/mapa'

def extraer_datos_anuncio(page,url):
    """
    Extrae el conjunto de datos que queremos de un anuncio en idealista accedido con ka variable page

    :param page: la pagina del anuncio que se va a analizar
    """

    vivienda = {
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
        vivienda['Nombre'] = info_primaria.ele('tag:span@class=main-info__title-main').text
        precio_loc = info_primaria.ele('tag:span@class=info-data-price')
        vivienda['Precio'] = precio_loc.ele('tag:span@class=txt-bold').text
        caract_box = page.ele('tag:section@class=details-box')
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



def extraer_certificado_ennergetico(info):
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
    

def extraer_imagenes(page,contenedor):

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
                    im.show()
                    lista.append({titulo.text:buffer_webp.getvalue()})
                    total+=1
        elif total >= 8:
            break
    return lista

def extrae_anunciante(info):
    contenedor = info.ele('tag:div@class=ide-box-contact module-contact-gray contact-data-container ')
    nombre = contenedor.ele('tag:div@class=professional-name').ele('tag:div@class=name')
    return nombre.text

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

    categoria = info.ele('tag:div@class=details-property-feature-one')
    if categoria:
        lista = categoria.eles('tag:li')
        for cat in lista:
            res = caracteristica(cat.text)
            for key,elem in res.items():
                if key in caracteristicas and caracteristicas[key] is None:
                    caracteristicas[key] = elem

    caracteristicas = {k:(False if v is None else v) for k,v in caracteristicas.items()}

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
    elif 'Planta' in texto or 'exterior' in texto or 'interior' in texto:
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
        res['Ventanas'] = ventana[0]
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

umbral = 1200

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
            if zona.ele('tag:a') and zona.ele('tag:a').attr('href').removesuffix('mapa') not in regiones_unicos:
                if int(zona.ele("tag:span@class=subdued").text.replace(".","")) < umbral :
                    link = zona.ele('tag:a').attr('href').removesuffix('mapa')
                    if "alquiler" in link :
                        region = link.removeprefix("https://www.idealista.com/alquiler-viviendas/madrid/")[:-1].replace('/',' - ')
                    else:
                        region = link.removeprefix("https://www.idealista.com/venta-viviendas/madrid/")[:-1].replace('/',' - ')
                    páginas_zonas.append({"region":region,"link":link,"num":int(zona.ele("tag:span@class=subdued").text.replace(".",""))})
                    regiones_unicos.add(zona.ele('tag:a').attr('href').removesuffix('mapa'))
                else:
                    zonas_expandir.append({"link":zona.ele('tag:a').attr('href'),"num":int(zona.ele("tag:span@class=subdued").text.replace(".",""))})
        print("Extrayendo links regiones restantes...")
        for urls in zonas_expandir:
            if urls not in páginas_zonas:
                page.get(urls["link"])
                páginas_zonas.extend(links_regiones(page,urls["link"],regiones_unicos,urls["num"]))

        return páginas_zonas


def sacar_link(anuncio,lista):
    tag_a = anuncio.ele('tag:a@role=heading')
    if tag_a:
        link = tag_a.attr('href')
        if link is not None and link not in lista:
            nombre = tag_a.attr('title')
            return {"nombre":nombre, "anuncio":link}
    return None

def analizar_pagina(page,lista_links):
    pagina = []
    anuncios = page.eles('tag:article')
    for anuncio in anuncios:
        clase = anuncio.attr('class')
        if 'adv' in clase:
            continue
        page.scroll.to_see(anuncio)
        anuncio_nuevo = sacar_link(anuncio,lista_links)
        if anuncio_nuevo:
            pagina.append(anuncio_nuevo)
            lista_links.add(anuncio_nuevo["anuncio"])
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


def obtiene_anuncios(links_regiones,page):
    anuncios_unicos = set()
    res = {}
    for url in links_regiones:
        siguiente = True
        page.get(url["link"])
        page = corrige_page(page,url["link"])
        res[url["region"]] = []
        barra_progreso = tqdm(desc=f"tratando la region de {url["region"]}")
        while siguiente:
            next = page.ele('.next')
            res[url["region"]].append(analizar_pagina(page,anuncios_unicos))
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
    print("\n   [A] 💰 VENTA")
    print("   [B] 🏠 ALQUILER")
    print("\n" + "─" * 60)

def imprimir_regiones(lista_datos):
    """
    lista_datos: [{'region': 'Madrid', 'num': 1500}, {'region': 'Chamberí', 'num': 250}]
    """
    total_anuncios = sum(item['num'] for item in lista_datos)
    
    print("\n" + "🗺️  MAPA DE EXTRACCIÓN ".center(56, "▒"))
    print(f"{'ID':<4} │ {'REGIÓN':<25} │ {'ANUNCIOS':>12}")
    print("─" * 56)

    for i, data in enumerate(lista_datos, 1):
        nombre = data['region']
        cantidad = f"{data['num']:,}".replace(",", ".")
        
        print(f"{i:<4} │ {nombre:<25} │ {cantidad:>10} 📢")

    print("─" * 56)
    
    total_str = f"{total_anuncios:,}".replace(",", ".")
    
    print("\n          ╔════════════════════════════════════════╗")
    print(f"          ║   [0]  🌟  PROCESAR TODAS LAS ZONAS    ║")
    print(f"          ║        Acumulado: {total_str:>12} ads      ║")
    print("          ╚════════════════════════════════════════╝\n")

    respuesta = int(input())
    while respuesta > len(lista_datos) and respuesta < 0:
        print("Entrada errónea")
        respuesta = input()
    

def webscraping_idealista():
    imprimir_header()
    imprimir_menu_modo()
    modo = input()
    while modo != 'A' and modo !=  'a' and modo != 'B' and modo != 'b':
        print("Error de entrada")
        imprimir_menu_modo()
        modo = input()
    if modo == 'A' or modo == 'a':
        url = url_venta
    elif modo == 'B' or modo == 'b':
        url = url_alquiler
    
    page = ChromiumPage()
    page.get(url)
    regiones_unicas = set()
    links_regiones_madrid = links_regiones(page,url,regiones_unicas,0)
    imprimir_regiones(links_regiones_madrid)
