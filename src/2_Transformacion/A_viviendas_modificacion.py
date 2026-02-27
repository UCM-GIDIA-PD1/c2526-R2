from src._1_Extraccion import A_Anuncios_viviendas
from DrissionPage import ChromiumPage
from geopy.geocoders import Nominatim

url_test='https://www.idealista.com/inmueble/107873687/'


page = ChromiumPage()
page.get(url_test)

if __name__=="__main__":
    vivienda = A_Anuncios_viviendas.extraer_datos_anuncio(page,url_test)
    geolocator = Nominatim(user_agent="maiday_scraper")

    # 2. La dirección que queremos buscar
    direccion = vivienda["Calle"] + ', Madrid, España'

    # 3. Hacemos la llamada
    localizacion = geolocator.geocode(direccion)

    # 4. Comprobamos si ha encontrado algo
    if localizacion:
        print(f"✅ Encontrado: {localizacion.address}")
        print(f"📍 Latitud: {localizacion.latitude}")
        print(f"📍 Longitud: {localizacion.longitude}")
        print(f"📦 Toda la info bruta: {localizacion.raw}")  # Aquí viene código postal, barrio, etc.
    else:
        print("❌ No se ha encontrado la dirección.")
