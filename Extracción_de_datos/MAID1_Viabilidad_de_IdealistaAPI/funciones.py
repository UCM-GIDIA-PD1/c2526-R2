import base64
import requests
import json
import urllib.parse
from httplib2 import Http

country = 'es'
language = 'es'
max_items= '50' #fijo como máximo por idealista
operation = 'sale'
order = 'distance' #la mejor estrategia si no queremos que las búsquedas solapen en los mismos anuncios
sort = 'asc' #ya que buscamos ordenando por distancia
multimedia = True # para que salgan anuncios que tienen imágenes

def codifica_acceso(api_Key,api_secret):
    apikey= urllib.parse.quote_plus(api_Key)
    secret= urllib.parse.quote_plus(api_secret)
    auth_str = f"{apikey}:{secret}"
    codificado = base64.b64encode(auth_str.encode('utf-8')).decode('utf-8')
    return codificado

def solicitar_token(credenciales,url):
    http_obj = Http()
    body = {'grant_type':'client_credentials'}
    headers = {'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8','Authorization' : 'Basic ' + credenciales}
    resp, content = http_obj.request(url,method='POST',headers=headers, body=urllib.parse.urlencode(body))
    if content:
        datos = json.loads(content)
        token_acceso = datos.get('access_token')
        tiempo = datos.get('expires_in')
        return {"token":token_acceso,"tiempo":tiempo}


def definirir_url(url_base,coord):
    res_url = (url_base + country + '/search?operation=' + operation + '&maxItems=' + max_items +'&propertyType' + proprety_type + '&locale=' + country + '&locationId=' + coord + '&order='+order+'&sort=' + sort + '&hasMultimedia' + multimedia)
    return res_url


def busca(url,acceso_token):
    headers = {'Content-Type': "application/json", 
               'Authorization' : 'Bearer ' + acceso_token}
    request = requests.post(url,headers=headers)
    anuncios = json.loads(requests.text)
    return anuncios
