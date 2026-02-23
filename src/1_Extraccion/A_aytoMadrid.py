import pandas as pd
import io
from funciones_minio import crear_cliente_minio, minio_subir_memoria


DATASETS = {
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
        "url":"https://datos.madrid.es/dataset/209548-0-censo-locales-historico/resource/209548-722-censo-locales-historico-csv/download/209548-722-censo-locales-historico-csv.csv",
        "object":"locales.parquet",
    },
    "PARQUES": {
        "url":"https://datos.madrid.es/dataset/200761-0-parques-jardines/resource/200761-0-parques-jardines-csv/download/200761-0-parques-jardines-csv.csv",
        "object":"parques.parquet",
    },
    "BIBLIOTECAS": {
        "url":"https://datos.madrid.es/egob/catalogo/201747-0-bibliobuses-bibliotecas.csv",
        "object": "bibliotecas.parquet",
    },
    "PARQUES_BOMBEROS": {
        "url":"https://datos.madrid.es/egob/catalogo/211642-0-bomberos-parques.csv",
        "object":"bomberos.parquet",
    },
    "CEMENTERIOS": {
        "url":"https://datos.madrid.es/egob/catalogo/205026-0-cementerios.csv",
        "object":"cementerios.parquet",
    },
    "CENTROS_DIA": {
        "url":"https://datos.madrid.es/egob/catalogo/200342-0-centros-dia.csv",
        "object":"centros_dia.parquet",
    },
    "COMISARIAS": {
        "url":"https://datos.madrid.es/egob/catalogo/300600-0-comisaria.csv",
        "object":"comisarias.parquet",
    },
    "POLIDEPORTIVOS": {
        "url":"https://datos.madrid.es/egob/catalogo/200186-0-polideportivos.csv",
        "object":"polideportivos.parquet",
    },
    "PUNTOS_LIMPIOS": {
        "url":"https://datos.madrid.es/egob/catalogo/200284-0-puntos-limpios-fijos.csv",
        "object":"puntos_limpios.parquet",
    },
    "IGLESIAS_CATOLICAS": {
        "url":"https://datos.madrid.es/egob/catalogo/209426-0-templos-catolicas.csv",
        "object":"iglesias.parquet",
    },
    "CENTROS_SERVICIOS_SOCIALES": {
        "url":"https://datos.madrid.es/egob/catalogo/209094-0-centros-servicios-sociales.csv",
        "object":"centros_sociales.parquet",
    },
    "CENTROS_MUNICIPALES_MAYORES": {
        "url":"https://datos.madrid.es/egob/catalogo/200337-0-centros-mayores.csv",
        "object": "centros_mayores.parquet",
    },
    "PISCINAS_MUNICIPALES": {
        "url":"https://datos.madrid.es/egob/catalogo/210227-0-piscinas-publicas.csv",
        "object":"piscinas.parquet",
    }
}

def preparar_parquet(url: str) -> io.BytesIO:
    """
    Lee un csv, lo guarda en memoria y lo convierte a parquet(en memoria)

    Args:
        url (str): url que descarga el csv

    Returns:
        io.BytesIO: buffer de memoria
    """
    df = pd.read_csv(url, sep=';', encoding= 'latin-1', dtype=str, low_memory=False)
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False)
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    carpeta = "raw"
    for nombre, cfg in DATASETS.items():
        buffer = preparar_parquet(cfg["url"])
        minio_subir_memoria(client=cliente, path=carpeta, buffer=buffer, minio_object=cfg["object"])
        print(f"OK: {nombre}")