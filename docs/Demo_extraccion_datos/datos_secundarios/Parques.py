import pandas as pd
import html
import re

# En este código se limpia el archivo (https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=dc758935dde13410VgnVCM2000000c205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default)
# Sacado del portal de datos abiertos del ayuntamiento de madrid 

archivo_limpieza = '200761-0-parques-jardines.csv'
df = pd.read_csv(archivo_limpieza, sep=';', encoding='latin-1')

cols_utiles = [
    'NOMBRE',
    'LATITUD',
    'LONGITUD',
    'BARRIO',
    'DISTRITO',
    'DESCRIPCION'
]

df_clean = df[cols_utiles].copy()

def limpiar_texto(texto):
    if pd.isna(texto): return ""
    return html.unescape(str(texto)).strip()

df_clean['NOMBRE'] = df_clean['NOMBRE'].apply(limpiar_texto)
df_clean['DESCRIPCION'] = df_clean['DESCRIPCION'].apply(limpiar_texto)

# Nos aseguramos q sean números. Nan los q fallen
df_clean['LATITUD'] = pd.to_numeric(df_clean['LATITUD'], errors='coerce')
df_clean['LONGITUD'] = pd.to_numeric(df_clean['LONGITUD'], errors='coerce')

# Eliminamos los q no tengan ubicación
df_clean = df_clean.dropna(subset=['LATITUD', 'LONGITUD'])

# Extraemos los m^2, pq un parque + grande puede que encarezca mas el precio.
def extraer_superficie(texto):
    # Busca el patrón "Superficie: [numero]"
    match = re.search(r'Superficie:\s*([\d\.]+)', texto)
    if match:
        return float(match.group(1).replace('.', ''))
    return None

df_clean['superficie_m2'] = df_clean['DESCRIPCION'].apply(extraer_superficie)
df_clean = df_clean.drop(columns=['DESCRIPCION'])

archivo_salida = 'parques_madrid_limpio.csv'
df_clean.to_csv(archivo_salida, index=False, encoding='utf-8')