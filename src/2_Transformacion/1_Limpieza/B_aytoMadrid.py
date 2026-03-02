import pandas as pd
from src.utils.funciones_minio import crear_cliente_minio, bajar_minio, subir_minio
from src.config import DATASETS_AYTO_LIMPIEZA, DISTRITOS,  MINIO_RAW_SECUNDARIOS, MINIO_INTERIM_SECUNDARIOS


import pandas as pd
from src.config import DISTRITOS

def limpiar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Nos quedamos solo con las columnas necesarias
    df = df[["PK", "LATITUD", "LONGITUD", "DISTRITO"]].copy()

    # Métricas antes de limpiar
    mask_nulos = df[["PK", "LATITUD", "LONGITUD", "DISTRITO"]].isna().any(axis=1)
    mask_inval = df["DISTRITO"].notna() & ~df["DISTRITO"].isin(DISTRITOS)

    # Filtro: eliminamos filas con nulos O con distrito inválido
    df = df.loc[~mask_nulos & ~mask_inval].copy()

    # Guardamos los distritos con su numeración oficial
    map_distritos = {d.upper(): i + 1 for i, d in enumerate(DISTRITOS)}
    df["NUM_DISTRITO"] = df["DISTRITO"].map(map_distritos).astype("int64")
    df = df.drop(columns=["DISTRITO"])

    # Cast seguro después de filtrar
    df = df.astype({
        "PK": "int64",
        "LATITUD": "float64",
        "LONGITUD": "float64"
    })
    
    return df

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    for nombre, obj in DATASETS_AYTO_LIMPIEZA.items():
        df = bajar_minio(client=cliente, path=MINIO_RAW_SECUNDARIOS, minio_object=obj)        
        df = limpiar_dataframe(df)
        subir_minio(df=df, client=cliente, path=MINIO_INTERIM_SECUNDARIOS, minio_object=obj)
        print(f"OK: {nombre}")
