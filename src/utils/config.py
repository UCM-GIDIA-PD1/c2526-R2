"""
config.py

Configuración centralizada del proyecto.
Reúne todas las constantes (URLs, rutas MinIO, nombres de ficheros, tags…)
que antes estaban dispersas en los scripts de extracción y transformación.
"""

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  RUTAS MinIO                                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

# --- embedings ---
MINIO_EMBEDDINGS = "dataset_ml"

# --- Extracción (raw) ---
MINIO_RAW = "raw"
MINIO_RAW_SECUNDARIOS = "raw/secundarios"
MINIO_RAW_PRIMARIOS = "raw/datos_primarios"

# --- Limpieza (cleaned) ---
MINIO_CLEANED_SECUNDARIOS = "cleaned/secundarios"
MINIO_CLEANED_TRANSPORTE = "cleaned/transporte"

# --- Procesado (processed) ---
MINIO_PROCESSED_SECUNDARIOS = "processed/secundarios"
PATH_DATASETS_MODELOS = "dataset_ml"

# --- Agrupacion (grouped) ---
MINIO_GROUPED_SECUNDARIOS = "grouped/secundarios"

# --- Datos secundarios temáticos ---
MINIO_CATASTRO = "cleaned/catastro"
MINIO_INE = "cleaned/ine"
MINIO_REJILLAS_SUCIO = "raw/rejillas"
MINIO_PADRON = "cleaned/padron"

# --- Datos primarios ---
MINIO_PRIMARIOS_IMAGENES = "datos_primarios/imagenes"
MINIO_PRIMARIOS_IMAGENES_ALQUILER = "datos_primarios/imagenes/alquiler"
MINIO_PRIMARIOS_IMAGENES_VENTA = "datos_primarios/imagenes/venta"

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  DISTRITOS DE MADRID                                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝


DISTRITOS = [
"CENTRO","ARGANZUELA","RETIRO","SALAMANCA","CHAMARTIN",
"TETUAN","CHAMBERI","FUENCARRAL-EL PARDO","MONCLOA-ARAVACA",
"LATINA","CARABANCHEL","USERA","PUENTE DE VALLECAS",
"MORATALAZ","CIUDAD LINEAL","HORTALEZA","VILLAVERDE",
"VILLA DE VALLECAS","VICALVARO","SAN BLAS-CANILLEJAS","BARAJAS"
]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  A — Idealista (Anuncios de viviendas)                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

URL_ALQUILER = "https://www.idealista.com/alquiler-viviendas/madrid-madrid/mapa"
URL_VENTA = "https://www.idealista.com/venta-viviendas/madrid-madrid/mapa"
IDEALISTA_UMBRAL_ANUNCIOS = 1200
CIUDAD = "Madrid"
PAIS = "Spain"
COLUMNAS_CARACTERISTICAS = ['id','Nombre','Barrio','Distrito','Calle','Precio','Superficie','Num_habitaciones','Banyos','Planta',
                      'Ventanas','Ascensor','Terraza','Balcon','Equipamiento','Cocina','Orientacion','Consumo','Descripcion','Anuncia','Url']

COLUMNAS_IMAGENES = ['id','Imagenes']
MODOS = ['venta','alquiler']
PATH_PRIMARIOS_LIMPIO = "datos_primarios"
PATH_PRIMARIOS_RAW = "raw/datos_primarios/"
ARCHIVOS_COORDENADAS = "coordenadas.parquet"
ARCHIVOS_VIVIENDAS = "viviendas"
ARCHIVOS_IMAGENES = "imagenes"



# ╔══════════════════════════════════════════════════════════════════════╗
# ║  B — Ayuntamiento de Madrid (datos abiertos)                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

DATASETS_AYTO = {
    "CENTROS_EDUCATIVOS": {
        "url": "https://datos.madrid.es/dataset/300614-0-centros-educativos/resource/300614-1-centros-educativos-csv/download/300614-1-centros-educativos-csv.csv",
        "object": "centros_educativos.parquet",
    },
    "UNIVERSIDADES": {
        "url": "https://datos.madrid.es/dataset/203166-0-universidades-educacion/resource/203166-0-universidades-educacion-csv/download/203166-0-universidades-educacion-csv.csv",
        "object": "universidades.parquet",
    },
    "HOSPITALES": {
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
        "object": "parques_bomberos.parquet",
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
    "IGLESIAS": {
        "url": "https://datos.madrid.es/egob/catalogo/209426-0-templos-catolicas.csv",
        "object": "iglesias.parquet",
    },
    "SERVICIOS_SOCIALES": {
        "url": "https://datos.madrid.es/egob/catalogo/209094-0-centros-servicios-sociales.csv",
        "object": "servicios_sociales.parquet",
    },
    "CENTROS_MAYORES": {
        "url": "https://datos.madrid.es/egob/catalogo/200337-0-centros-mayores.csv",
        "object": "centros_mayores.parquet",
    },
    "PISCINAS": {
        "url": "https://datos.madrid.es/egob/catalogo/210227-0-piscinas-publicas.csv",
        "object": "piscinas.parquet",
    },
}

# Subconjunto de datasets usados en la fase de limpieza (excluye LOCALES)
DATASETS_AYTO_LIMPIEZA = {k: v["object"] for k, v in DATASETS_AYTO.items() if k != "LOCALES"}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  C — Transporte público (ArcGIS / CRTM)                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

URL_BUS = (
    "https://services5.arcgis.com/UxADft6QPcvFyDU1/arcgis/rest/services/"
    "M6_Red/FeatureServer/0/query?where=1%3D1"
    "&outFields=*&outSR=4326&f=json"
)

URL_METRO_BASE = (
    "https://services5.arcgis.com/UxADft6QPcvFyDU1/arcgis/rest/services/"
    "Lineas_Metro/FeatureServer"
)

# IDs de las capas de ESTACION (sentido 1 = S1, una por línea de metro)
# y nombre de línea asociado a cada capa
METRO_LAYER_IDS = [2, 11, 20, 29, 38, 47, 56, 59, 71, 80, 83, 95, 98, 110, 119, 128]
METRO_LAYER_LINEAS = {
    2: "1", 11: "2", 20: "3", 29: "4", 38: "5", 47: "6",
    56: "7a", 59: "7b", 71: "8", 80: "9A", 83: "9B",
    95: "10a", 98: "10b", 110: "11", 119: "12", 128: "R",
}

OBJ_BUS = "paradas_bus.parquet"
OBJ_METRO = "estaciones_metro.parquet"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  D — Catastro                                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

URL_CATASTRO = "https://www.catastro.hacienda.gob.es/INSPIRE/Buildings/28/28900-MADRID/A.ES.SDGC.BU.28900.zip"
OBJ_CATASTRO = "anio_construccion.parquet"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  E — INE (renta media por hogar)                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

URL_INE = "https://servicios.ine.es/wstempus/js/es/DATOS_TABLA/30824?tip=AM"
OBJ_INE = "renta_hogar_secciones_madrid.parquet"
OBJ_INE_JUNTO = "renta_media.parquet"

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

OBJ_COMERCIO = "comercios.parquet"
OBJ_NEGATIVOS = "negativos.parquet"
OBJ_ALIMENTACION = "alimentacion.parquet"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  G — Capas (Secciones censales, Barrios, Padron)                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
URL_PADRON = "https://datos.madrid.es/dataset/200076-0-padron/resource/200076-1-padron-json/download/200076-1-padron-json.json"
URL_BARRIOS = "https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Barrios.zip"
URL_SECCIONES = "https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Seccionado/Secciones_Censales.zip"
OBJ_PADRON_BAR = "padron_barrio_madrid.parquet"
OBJ_PADRON_SEC = "padron_seccion_madrid.parquet"
OBJ_SECCIONES = "secciones_censales_madrid.parquet"
OBJ_BARRIOS = "barrios_madrid.parquet"
TIPOS_REJILLAS = [{"tipo":"barrios","columna_id":"COD_BAR"},{"tipo":"secciones censales","columna_id":"CUSEC"},{"tipo":"hexagonos_1","columna_id":"id_hex"},{"tipo":"hexagonos_2","columna_id":"id_hex"}]
COD_REJILLA = {"barrios":"COD_BAR","secciones censales":"CUSEC","hexagonos_1":"id_hex","hexagonos_2":"id_hex"}
COMPONENTES_TRANSPORTE = [{"tipo":"bus","fichero":"paradas_bus.parquet","calculo":"paradas"},{"tipo":"metro","fichero":"estaciones_metro.parquet","calculo":"estaciones"}]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  H — Modelo CNN (Clasificación de imágenes)                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

CLASES_IMAGENES = ['Cocina', 'Dormitorio', 'Salón', 'Banyo']
MINIO_DATASET_VISION = "cleaned/dataset_vision"
CNN_TARGET_SIZE = (150, 150)
CNN_BATCH_SIZE = 32
