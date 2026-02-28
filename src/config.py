"""
config.py

Configuración centralizada del proyecto.
Reúne todas las constantes (URLs, rutas MinIO, nombres de ficheros, tags…)
que antes estaban dispersas en los scripts de extracción y transformación.
"""

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  RUTAS MinIO                                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

# --- Extracción (raw) ---
MINIO_RAW = "raw"
MINIO_RAW_SECUNDARIOS = "raw/secundarios"
MINIO_RAW_PRIMARIOS = "raw/datos_primarios"

# --- Transformación (interim) ---
MINIO_INTERIM_SECUNDARIOS = "interim/secundarios"

# --- Datos secundarios temáticos ---
MINIO_CATASTRO = "datos_secundarios/catastro"
MINIO_INE = "datos_secundarios/ine"
MINIO_COMERCIO = "datos_secundarios/comercio"
MINIO_NEGATIVOS = "datos_secundarios/negativos"
MINIO_SECCIONES = "datos_secundarios/secciones_censales"
MINIO_ALIMENTACION = "datos_secundarios/alimentacion"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  A — Idealista (Anuncios de viviendas)                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

URL_ALQUILER = "https://www.idealista.com/alquiler-viviendas/madrid-madrid/mapa"
URL_VENTA = "https://www.idealista.com/venta-viviendas/madrid-madrid/mapa"
IDEALISTA_UMBRAL_ANUNCIOS = 1200
CIUDAD = "Madrid"
PAIS = "Spain"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  B — Ayuntamiento de Madrid (datos abiertos)                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

DATASETS_AYTO = {
    "EDUCACION": {
        "url": "https://datos.madrid.es/dataset/300614-0-centros-educativos/resource/300614-1-centros-educativos-csv/download/300614-1-centros-educativos-csv.csv",
        "object": "centros_educativos.parquet",
    },
    "UNIVERSIDAD": {
        "url": "https://datos.madrid.es/dataset/203166-0-universidades-educacion/resource/203166-0-universidades-educacion-csv/download/203166-0-universidades-educacion-csv.csv",
        "object": "universidades.parquet",
    },
    "SANIDAD": {
        "url": "https://datos.madrid.es/dataset/212769-0-atencion-medica/resource/212769-0-atencion-medica-csv/download/212769-0-atencion-medica-csv.csv",
        "object": "hospitales.parquet",
    },
    "LOCALES": {
        "url": "https://datos.madrid.es/dataset/209548-0-censo-locales-historico/resource/209548-722-censo-locales-historico-csv/download/209548-722-censo-locales-historico-csv.csv",
        "object": "locales.parquet",
    },
    "PARQUES": {
        "url": "https://datos.madrid.es/dataset/200761-0-parques-jardines/resource/200761-0-parques-jardines-csv/download/200761-0-parques-jardines-csv.csv",
        "object": "parques.parquet",
    },
    "BIBLIOTECAS": {
        "url": "https://datos.madrid.es/egob/catalogo/201747-0-bibliobuses-bibliotecas.csv",
        "object": "bibliotecas.parquet",
    },
    "PARQUES_BOMBEROS": {
        "url": "https://datos.madrid.es/egob/catalogo/211642-0-bomberos-parques.csv",
        "object": "bomberos.parquet",
    },
    "CEMENTERIOS": {
        "url": "https://datos.madrid.es/egob/catalogo/205026-0-cementerios.csv",
        "object": "cementerios.parquet",
    },
    "CENTROS_DIA": {
        "url": "https://datos.madrid.es/egob/catalogo/200342-0-centros-dia.csv",
        "object": "centros_dia.parquet",
    },
    "COMISARIAS": {
        "url": "https://datos.madrid.es/egob/catalogo/300600-0-comisaria.csv",
        "object": "comisarias.parquet",
    },
    "POLIDEPORTIVOS": {
        "url": "https://datos.madrid.es/egob/catalogo/200186-0-polideportivos.csv",
        "object": "polideportivos.parquet",
    },
    "PUNTOS_LIMPIOS": {
        "url": "https://datos.madrid.es/egob/catalogo/200284-0-puntos-limpios-fijos.csv",
        "object": "puntos_limpios.parquet",
    },
    "IGLESIAS_CATOLICAS": {
        "url": "https://datos.madrid.es/egob/catalogo/209426-0-templos-catolicas.csv",
        "object": "iglesias.parquet",
    },
    "CENTROS_SERVICIOS_SOCIALES": {
        "url": "https://datos.madrid.es/egob/catalogo/209094-0-centros-servicios-sociales.csv",
        "object": "centros_sociales.parquet",
    },
    "CENTROS_MUNICIPALES_MAYORES": {
        "url": "https://datos.madrid.es/egob/catalogo/200337-0-centros-mayores.csv",
        "object": "centros_mayores.parquet",
    },
    "PISCINAS_MUNICIPALES": {
        "url": "https://datos.madrid.es/egob/catalogo/210227-0-piscinas-publicas.csv",
        "object": "piscinas.parquet",
    },
}

# Subconjunto de datasets usados en la fase de limpieza (excluye LOCALES)
DATASETS_AYTO_LIMPIEZA = {k: v["object"] for k, v in DATASETS_AYTO.items() if k != "LOCALES"}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  C — Transporte público (ArcGIS / CRTM)                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

URL_BUS = (
    "https://services5.arcgis.com/UxADft6QPcvFyDU1/arcgis/rest/services/"
    "M6_Red/FeatureServer/0/query?where=1%3D1"
    "&outFields=DENOMINACION,X,Y,GRADOACCESIBILIDAD&outSR=4326&f=json"
)

URL_METRO_BASE = (
    "https://services5.arcgis.com/UxADft6QPcvFyDU1/arcgis/rest/services/"
    "Lineas_Metro/FeatureServer"
)

# IDs de las capas de ESTACION (sentido 1 = S1, una por línea de metro)
METRO_LAYER_IDS = [2, 11, 20, 29, 38, 47, 56, 59, 71, 80, 83, 95, 98, 110, 119, 128]

OBJ_BUS = "paradas_bus.parquet"
OBJ_METRO = "estaciones_metro.parquet"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  D — Catastro                                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

URL_CATASTRO = "https://www.catastro.hacienda.gob.es/INSPIRE/Buildings/28/28900-MADRID/A.ES.SDGC.BU.28900.zip"
OBJ_CATASTRO = "edificios_madrid.parquet"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  E — INE (renta media por hogar)                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

URL_INE = "https://servicios.ine.es/wstempus/js/es/DATOS_TABLA/30824?tip=AM"
OBJ_INE = "renta_hogar_secciones_madrid.parquet"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  F — OpenStreetMap (ocio, negativos y supermercados y comercios)     ║
# ╚══════════════════════════════════════════════════════════════════════╝

PLACE_OSM = "Madrid, Spain"

TAGS_COMERCIO = {
    "shop": ["mall", "department_store", "clothes", "electronics", "hairdresser", "beauty", "shoes", "hardware"],
    "amenity": ["restaurant", "cafe", "bar", "pub", "cinema", "pharmacy", "veterinary"]
}

TAGS_NEGATIVOS = {
    "landuse": ["industrial", "landfill"],
    "amenity": ["prison", "grave_yard"],
    "power": ["substation"],
}

TAGS_ALIMENTACION = {
    "shop": ["supermarket", "convenience", "bakery", "butcher", "greengrocer", "seafood"],
    "amenity": ["marketplace", "market"]
}

OBJ_COMERCIO = "indicadores_comercio_madrid.parquet"
OBJ_NEGATIVOS = "indicadores_negativos_madrid.parquet"
OBJ_ALIMENTACION = "indicadores_alimentacion_madrid.parquet"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  G — Secciones censales (Geoportal Ayto. Madrid)                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

URL_SECCIONES = "https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Seccionado/TopoJSON/Secciones_Censales.json"
OBJ_SECCIONES = "secciones_censales_madrid.parquet"
