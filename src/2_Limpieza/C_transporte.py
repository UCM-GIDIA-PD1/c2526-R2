import pandas as pd
from shapely import wkb
from utils.funciones_minio import crear_cliente_minio, bajar_minio, subir_minio
from utils.config import MINIO_CLEANED_TRANSPORTE, MINIO_PROCESSED_SECUNDARIOS, MINIO_RAW_SECUNDARIOS, OBJ_BUS, OBJ_METRO


def limpiar_transporte(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """Selecciona y renombra las columnas de interés del dataset de transporte.

    Columnas resultantes: nombre, lat, lon, lineas

    - DENOMINACION → nombre (nombre de la estación de metro / parada de bus)
    - LINEAS  → lineas (lista de líneas)
    - geometry.y → lat
    - geometry.x → lon
    """
    out = pd.DataFrame({
        "nombre": df["DENOMINACION"],
        "lat": df["geometry"].apply(lambda geom: wkb.loads(geom).y),
        "lon": df["geometry"].apply(lambda geom: wkb.loads(geom).x),
        "lineas": df["LINEAS"],
    })

    # Normalizar nombre y eliminar registros sin nombre
    out = out.dropna(subset=["nombre"])
    out["nombre"] = out["nombre"].astype(str).str.strip().str.upper()
    out = out[out["nombre"] != ""]

    # Convertir 'lineas' de string (ej. "4, 5, 10") a lista de strings
    if not out.empty:
        out["lineas"] = out["lineas"].astype(str).str.split(",").apply(lambda x: [s.strip() for s in x if s.strip()])

    # Eliminar duplicados (la nueva API de metro tiene múltiples entradas por transbordo pero con la misma info)
    out = out.drop_duplicates(subset=["nombre"], keep='first').reset_index(drop=True)

    print(f"  {nombre}: {len(out)} registros únicos → columnas {list(out.columns)}")
    return out


if __name__ == "__main__":
    cliente = crear_cliente_minio()

    for nombre, obj in {"BUS": OBJ_BUS, "METRO": OBJ_METRO}.items():
        df = bajar_minio(client=cliente, path=MINIO_RAW_SECUNDARIOS, minio_object=obj)
        df = limpiar_transporte(df, nombre)
        subir_minio(df=df, client=cliente, path=MINIO_CLEANED_TRANSPORTE, minio_object=obj)
        print(f"OK: {nombre}")
