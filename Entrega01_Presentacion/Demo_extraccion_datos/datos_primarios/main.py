import random
import os
import time
from DrissionPage import ChromiumPage
from pathlib import Path
import pandas as pd
from funciones import *

ruta_original = Path(__file__).resolve().parent

ruta_archivo = ruta_original / 'datos_idealista'

ruta_csv_anuncios = ruta_archivo / 'links_anuncios.csv'
ruta_csv_regiones = ruta_archivo / 'regiones.csv'

os.makedirs(ruta_archivo, exist_ok=True)
page = ChromiumPage()

url_madrid = "https://www.idealista.com/alquiler-viviendas/madrid-madrid/mapa"

page.get(url_madrid)

regiones_unicas = set()

links_zonas = links_regiones(page,regiones_unicas,0)
guardas_links_regiones(links_zonas,ruta_csv_regiones)

anuncios_unicos = set()

for url in links_zonas:
    región = url["link"].removeprefix('https://www.idealista.com/alquiler-viviendas/madrid/')
    lista = []
    siguiente = True
    cont = 1
    page.get(url["link"])
    while siguiente:
        next = page.ele('.next')
        guardar_pagina_en_csv(analizar_pagina(page,anuncios_unicos), ruta_csv_anuncios)
        if not next:
            siguiente = False
        else:
            url_next = next.ele('tag:a').attr('href')
            page.get(url_next)
        cont+=1
        time.sleep(random.uniform(5, 10))



