import pandas as pd
from shapely import wkb
from src.utils.funciones_minio import crear_cliente_minio, bajar_minio, subir_minio
from src.config import MINIO_RAW_SECUNDARIOS, MINIO_INTERIM_SECUNDARIOS, OBJ_BUS, OBJ_METRO


def limpiar_transporte(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """Selecciona y renombra las columnas de interés del dataset de transporte.

    Columnas resultantes: lin, lat, lon.

    - LINEAS  → lin  (líneas que pasan por la parada / estación)
    - Latitud → lat  (Y en bus, geometry.y en metro)
    - Longitud → lon (X en bus, geometry.x en metro)
    """
    # Si el dataset tiene columnas X / Y explícitas (bus), usarlas directamente
    if "X" in df.columns and "Y" in df.columns:
        out = df[["LINEAS", "Y", "X"]].copy()
        out.columns = ["lin", "lat", "lon"]
    # Si solo tiene geometría (metro), extraer coordenadas de la columna geometry
    elif "geometry" in df.columns:
        out = pd.DataFrame({
            "lin": df["LINEAS"],
            "lat": df["geometry"].apply(lambda geom: wkb.loads(geom).y),
            "lon": df["geometry"].apply(lambda geom: wkb.loads(geom).x),
        })
    else:
        raise ValueError(f"No se encontraron columnas de coordenadas en {nombre}")

    print(f"  {nombre}: {len(out)} registros → columnas {list(out.columns)}")
    return out


if __name__ == "__main__":
    cliente = crear_cliente_minio()

    for nombre, obj in {"BUS": OBJ_BUS, "METRO": OBJ_METRO}.items():
        df = bajar_minio(client=cliente, path=MINIO_RAW_SECUNDARIOS, minio_object=obj)
        df = limpiar_transporte(df, nombre)
        subir_minio(df=df, client=cliente, path=MINIO_INTERIM_SECUNDARIOS, minio_object=obj)
        print(f"OK: {nombre}")
