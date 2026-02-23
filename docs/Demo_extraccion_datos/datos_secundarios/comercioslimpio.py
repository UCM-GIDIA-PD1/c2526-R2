#librerias necesarias
import requests
import re
from pathlib import Path
import pandas as pd
import numpy as np

ruta_original = Path(__file__).resolve().parent
ruta_carpeta = ruta_original / "comercios" / "comercios_madrid" 
ruta_carpeta.mkdir(parents=True,exist_ok=True) #crea la carpeta si no existe

ruta_csv = ruta_carpeta / "comercios.csv"
url_api = "https://datos.madrid.es/egob/catalogo/209548-802-censo-locales-historico.csv" #url de la API

response = requests.get(url_api,stream=True) #stream=True lee un flujo de datos poco a poco para que tarde menos en ejecutarse
response.raise_for_status() #comprobar codigo HTTP
with open(ruta_csv,"wb") as f:
  for chunk in response.iter_content(chunk_size=1024*1024): #descargarlo en local de 1mb en 1mb
    f.write(chunk)

#meterlo en un dataframe
df = pd.read_csv(ruta_csv,sep = ";", encoding= "latin-1")

cols_mantenidas = ["ï»¿id_local","id_distrito_local","desc_distrito_local","coordenada_x_local","coordenada_y_local","desc_situacion_local","desc_vial_edificio","clase_vial_acceso","num_acceso","cod_postal"]
df_a_limpiar = df[cols_mantenidas].copy()

df = df_a_limpiar.copy()
df.rename(columns={'ï»¿id_local': 'id_local'}, inplace=True) #renombrar columna

df = df.astype(str) #convertir todas las columnas a string para quitar todos los espacios en blanco
df = df.apply(lambda x: x.str.strip())

#volver a convertir las columnas que son numeros a su tipo original (int o float)
cols_enteros = ["id_local","id_distrito_local","num_acceso","cod_postal"]
for col in cols_enteros:
  df[col] = pd.to_numeric(df[col],errors = "coerce").astype("Int64")

cols_float = ["coordenada_x_local","coordenada_y_local"]
for col in cols_float:
  df[col] = pd.to_numeric(df[col],errors="coerce").astype(float)

df_2 = df.copy()
df_2 = df_2[~((df_2["coordenada_x_local"]==0) & (df_2["coordenada_y_local"]==0))] #quitar las filas donde las coordenadas x e y sean 0
df = df_2 

df2 = df.copy()
df2["desc_situacion_local"] = np.where(df2["desc_situacion_local"] == "Abierto",1,0) #convertir la columna desc_situacion_local a 1 y 0 (1=Abierto, 0=Cerrado)

df_limpio = df2.copy() #ordenar el dataframe
df_limpio = df2[["id_local","id_distrito_local","desc_distrito_local","clase_vial_acceso","desc_vial_edificio","num_acceso",
                 "coordenada_x_local","coordenada_y_local","cod_postal","desc_situacion_local"]]
df_limpio.reset_index(drop=True,inplace=True) #resetear el indice del dataframe


#guardar el dataframe limpio junto con el head en ficheros .csv
partes = url_api.rstrip('/').split('/')[-1].split('-')
nombre_fichero = f"comercios_limpios_" + partes[-1]
nombre_fichero_head = f"comercios_limpios_head" + partes[-1]
ruta = ruta_carpeta / nombre_fichero
ruta_head = ruta_carpeta / nombre_fichero_head
df_limpio_head = df_limpio.copy()
df_limpio_head = df_limpio_head.head()
df_limpio.to_csv(ruta,index=False,encoding="utf-8-sig")
df_limpio_head.to_csv(ruta_head,index = False,encoding="utf-8-sig")