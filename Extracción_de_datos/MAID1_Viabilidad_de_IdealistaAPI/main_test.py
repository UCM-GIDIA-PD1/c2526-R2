import pandas as pd
from path import Path
import base64
from funciones import *

url_base_api = 'https://api.idealista.com/3.5' 

coordenadas = ["40.416789,-3.703517"]     #   coordenadas de la puerta de sol por ejemplo

api_key = "" # nuestra apiKey
api_secret = "" # nuestra apisecret

credenciales = codifica_acceso(api_key,api_secret)

url_token = "https://api.idealista.com/oauth/token"

token_acceso = solicitar_token(credenciales,url_token)

print(f"tienes : {token_acceso["tiempo"]} secundos antes que caduzca el acceso, con token {token_acceso["token"]}")

"""
df = pd.DataFrame()

for zona in coordenadas:
    url = definirir_url(url_base_api,coordenadas)
    lista_pisos_zonas = busca(url,token_acceso)
    df_new = pd.DataFrame.from_dict(resultas['elementList'])
    df = pd.concat(df,df_new)

ruta = Path(__file__).resolve().parent

ruta_archivo = ruta / 'anuncios_venta_api.csv'

df.to_csv(ruta_archivo,mode='a',index=False,encoding = 'utf-8')


"""






