from DrissionPage import ChromiumPage
import time
import pandas as pd
import os

umbral = 1200

def links_regiones(page,regiones_unicos,num = 0):
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
            time.sleep(2)
            contenedor = page.ele('.navList nav-list')
        else:
            contenedor = page.ele('.navList')
        lista_zonas = contenedor.eles('tag:li')
        for zona in lista_zonas :
            if zona.ele('tag:a') and zona.ele('tag:a').attr('href').removesuffix('mapa') not in regiones_unicos:
                if int(zona.ele("tag:span@class=subdued").text.replace(".","")) < umbral :
                    páginas_zonas.append({"link":zona.ele('tag:a').attr('href').removesuffix('mapa'),"num":int(zona.ele("tag:span@class=subdued").text.replace(".",""))})
                    regiones_unicos.add(zona.ele('tag:a').attr('href').removesuffix('mapa'))
                else:
                    zonas_expandir.append({"link":zona.ele('tag:a').attr('href'),"num":int(zona.ele("tag:span@class=subdued").text.replace(".",""))})
        
        for urls in zonas_expandir:
            if urls not in páginas_zonas:
                page.get(urls["link"])
                páginas_zonas.extend(links_regiones(page,regiones_unicos,urls["num"]))

        return páginas_zonas



def sacar_link(anuncio,lista):
    tag_a = anuncio.ele('tag:a@role=heading')
    link = tag_a.attr('href')
    if link not in lista:
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