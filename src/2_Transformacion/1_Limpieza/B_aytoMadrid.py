import pandas as pd
from src.utils.funciones_minio import crear_cliente_minio, bajar_minio, subir_minio
from src.config import DATASETS_AYTO_LIMPIEZA, MINIO_RAW_SECUNDARIOS, MINIO_INTERIM_SECUNDARIOS


if __name__ == "__main__":
    cliente = crear_cliente_minio()
    for nombre, obj in DATASETS_AYTO_LIMPIEZA.items():
        df = bajar_minio(client=cliente, path=MINIO_RAW_SECUNDARIOS, minio_object=obj)        
        df = df[["PK", "NOMBRE", "TRANSPORTE", "NOMBRE-VIA", 
                "NUM", "LATITUD", "LONGITUD", "ACCESIBILIDAD"]]
        subir_minio(df=df, client=cliente, path=MINIO_INTERIM_SECUNDARIOS, minio_object=obj)
        print(f"OK: {nombre}")

