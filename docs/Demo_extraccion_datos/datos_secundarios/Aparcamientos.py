import requests
import re
from pathlib import Path
import pandas as pd

ruta_original = Path(__file__).resolve().parent

ruta_carpeta = ruta_original / 'datos_secundarios' /  'aparcamientos_madrid_interesantes'

ruta_carpeta.mkdir(parents=True,exist_ok=True)

urls_api = ['https://datos.madrid.es/egob/catalogo/202584-0-aparcamientos-residentes.csv','https://datos.madrid.es/egob/catalogo/202625-0-aparcamientos-publicos.csv']

def busca_num(contenido,pos):
    if pos >= len(contenido):
        return -1
    
    while pos < len(contenido) and not contenido[pos].replace('.','').isdigit() :
        contenido[pos] = contenido[pos].replace('.','')
        pos+=1
    
    if pos < len(contenido):
        contenido[pos] = contenido[pos].replace('.','')
        return pos

def desentramar(contenido):
    nuevo_contenido = []
    borrando = False
    for i in range(0,len(contenido)):
        if i <  len(contenido):
            partes = re.findall(r'(\d+)([a-zA-Z]+)', contenido[i])
            if partes:
                for parte in partes:
                    for argumento in parte:
                        nuevo_contenido.append(argumento)
            elif "(" in contenido[i]:
                borrando = True
            elif ")" in contenido[i]:
                borrando = False
            elif not borrando:
                nuevo_contenido.append(contenido[i])

    return nuevo_contenido

def extraccion_plazas(fila):

    plazas={
        'TOTAL':None,
        'RESIDENCIAL':None,
        'PUBLICO':None
    }
    if pd.isna(fila['DESCRIPCION']):
        if pd.isna(fila['DESCRIPCION-ENTIDAD']):
            return None
        contenido = fila['DESCRIPCION-ENTIDAD'].split()
    else:
        contenido = fila['DESCRIPCION'].split()
    contenido = desentramar(contenido)
    pos = 0
    if  isinstance(contenido[pos+1],int):
        pos+=1
    while plazas['TOTAL'] is None and ( plazas['RESIDENCIAL'] is None or plazas['PUBLICO'] is None):
        pos = busca_num(contenido,pos)
        if  plazas['PUBLICO'] is None and (contenido[pos + 1].lower() == 'públicas' or contenido[pos + 1].lower() == 'rotacionales'):
            plazas['PUBLICO'] = int(contenido[pos])
            pos+=1
            if plazas['RESIDENCIAL'] is None:
                pos = busca_num(contenido,pos)
        elif plazas['RESIDENCIAL'] is None and (contenido[pos+1].lower() == 'residenciales' or contenido[pos+2].lower().replace('.','') == 'residentes' or contenido[pos+1].lower() == 'residentes'):
            plazas['RESIDENCIAL'] = int(contenido[pos])
            pos+=1
            if plazas['PUBLICO'] is None:
                pos = busca_num(contenido,pos)
        elif contenido[pos+1].replace(':','').lower() == 'abierto' or contenido[pos+1].lower().replace(',','') == 'plazas' or contenido[pos+1].replace(':','').lower() == 'titularidad' or contenido[pos+1].replace(':','').lower() == 'información' or contenido[pos-1].lower() == 'automóviles:' or contenido[pos+1].lower() == 'actualmente':
            plazas['TOTAL'] = int(contenido[pos])
            return plazas
        else:
            pos+=1
            pos = busca_num(contenido,pos)
    if not plazas['TOTAL']:
        plazas['TOTAL'] = plazas['RESIDENCIAL'] + plazas['PUBLICO']

    return plazas

for url in urls_api:
    response = requests.get(url)

    if response.status_code == 200:
        df = pd.read_csv(url,sep = ";",encoding = 'latin-1')
    else:
        print("ERROR")

    

    partes = url.rstrip('/').split('/')[-1].split('-')
    nombre_fichero = f"distribucion-{'-'.join(partes[-2:])}"

    ruta =  ruta_carpeta / nombre_fichero

    df_nuevas = df.apply(extraccion_plazas, axis=1, result_type='expand')

    df = pd.concat([df, df_nuevas], axis=1)

    columnas_a_mantener = ['PK','NOMBRE', 'BARRIO', 'DISTRITO','COORDENADA-X','COORDENADA-Y', 'LATITUD', 'LONGITUD', 'TOTAL', 'RESIDENCIAL', 'PUBLICO']

    df_final = df[columnas_a_mantener].copy()

    df_final.to_csv(ruta,index=False, encoding='utf-8-sig')


