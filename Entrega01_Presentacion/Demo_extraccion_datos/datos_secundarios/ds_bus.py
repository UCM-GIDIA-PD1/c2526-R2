import requests
import pandas as pd
import json
import geopandas as gpd

def descargar_datos_raw():
    # La URL filtrada en formato GeoJSON si quiero otro formato cambiar f=json al formato que quiero y si quiero todas las columnas poner outFields=*
    url = "https://services5.arcgis.com/UxADft6QPcvFyDU1/arcgis/rest/services/M6_Red/FeatureServer/0/query?where=1%3D1&outFields=DENOMINACION,X,Y,GRADOACCESIBILIDAD&outSR=4326&f=json"

    print("Conectando con la API del CRTM...")
    response = requests.get(url)

    if response.status_code == 200:
        # Convertimos el JSON en una lista de diccionarios
        data = response.json()
        features = [f['attributes'] for f in data['features']]

        # Creamos el DataFrame y lo guardamos sin tocar nada (RAW)
        df_raw = pd.DataFrame(features)
        df_raw.to_csv("bus_madrid_raw.csv", index=False)
        print("✓ Archivo 'bus_madrid_raw.csv' guardado correctamente.")
    else:
        print(f"Error al descargar: {response.status_code}")


if __name__ == "__main__":
    descargar_datos_raw()
