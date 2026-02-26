import pandas as pd
from src.utils.funciones_minio import crear_cliente_minio, bajar_minio, subir_minio


# IMPORTANTE: NO SE UTILIZAN AHORA LOS DATOS DE LOCALES

DATASETS = {
    "EDUCACION": "centros_educativos.parquet",
    "UNIVERSIDAD": "universidades.parquet",
    "SANIDAD": "hospitales.parquet",
    #"LOCALES": "locales.parquet",
    "PARQUES": "parques.parquet",
    "BIBLIOTECAS": "bibliotecas.parquet",
    "PARQUES_BOMBEROS": "bomberos.parquet",
    "CEMENTERIOS": "cementerios.parquet",
    "CENTROS_DIA": "centros_dia.parquet",
    "COMISARIAS": "comisarias.parquet",
    "POLIDEPORTIVOS": "polideportivos.parquet",
    "PUNTOS_LIMPIOS": "puntos_limpios.parquet",
    "IGLESIAS_CATOLICAS": "iglesias.parquet",
    "CENTROS_SERVICIOS_SOCIALES": "centros_sociales.parquet",
    "CENTROS_MUNICIPALES_MAYORES": "centros_mayores.parquet",
    "PISCINAS_MUNICIPALES": "piscinas.parquet",
}

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    carpeta_raw = "raw/secundarios"
    carpeta_destino = "interim/secundarios"
    for nombre, obj in DATASETS.items():
        df = bajar_minio(client=cliente, path=carpeta_raw, minio_object=obj)        
        df = df[["PK", "NOMBRE", "TRANSPORTE", "NOMBRE-VIA", 
                "NUM", "LATITUD", "LONGITUD", "ACCESIBILIDAD"]]
        subir_minio(df=df, client=cliente, path=carpeta_destino, minio_object=obj)
        print(f"OK: {nombre}")
