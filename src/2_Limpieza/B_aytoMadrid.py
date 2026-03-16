import pandas as pd
from utils.funciones_minio import crear_cliente_minio, bajar_minio, subir_minio
from utils.config import DATASETS_AYTO_LIMPIEZA, DISTRITOS,  MINIO_RAW_SECUNDARIOS, MINIO_CLEANED_SECUNDARIOS

def limpiar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y normaliza un DataFrame del Ayuntamiento de Madrid.

    Selecciona columnas relevantes, elimina registros inválidos y
    genera la codificación numérica oficial del distrito.

    Args:
        df (pd.DataFrame): DataFrame original con al menos las columnas
            ["PK", "LATITUD", "LONGITUD", "DISTRITO"].

    Returns:
        pd.DataFrame: DataFrame limpio con el siguiente esquema:
            - nombre (string)
            - lat (float64)
            - lon (float64)
    """
    # Nos quedamos solo con las columnas necesarias
    df = df[["NOMBRE", "LATITUD", "LONGITUD", "DISTRITO"]].copy()

    # Métricas antes de limpiar
    mask_nulos = df[["NOMBRE", "LATITUD", "LONGITUD", "DISTRITO"]].isna().any(axis=1)
    mask_inval = df["DISTRITO"].notna() & ~df["DISTRITO"].isin(DISTRITOS)

    # Filtro: eliminamos filas con nulos O con distrito inválido
    df = df.loc[~mask_nulos & ~mask_inval].copy()
    df = df.drop(columns=["DISTRITO"])

    # Cast seguro después de filtrar
    df = df.astype({
        "NOMBRE": "string",
        "LATITUD": "float64",
        "LONGITUD": "float64"
    })

    # Cambiar nombre columnas
    df.columns = ["nombre", "lat", "lon"]
    return df

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    for nombre, obj in DATASETS_AYTO_LIMPIEZA.items():
        df = bajar_minio(client=cliente, path=MINIO_RAW_SECUNDARIOS, minio_object=obj)        
        df = limpiar_dataframe(df)
        subir_minio(df=df, client=cliente, path=MINIO_CLEANED_SECUNDARIOS, minio_object=obj)
        print(f"OK: {nombre}")
