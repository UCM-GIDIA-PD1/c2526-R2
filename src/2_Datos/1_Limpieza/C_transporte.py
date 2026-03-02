import pandas as pd
from shapely import wkb
from src.utils.funciones_minio import crear_cliente_minio, bajar_minio, subir_minio
from src.config import MINIO_RAW_SECUNDARIOS, MINIO_CLEANED_SECUNDARIOS, OBJ_BUS, OBJ_METRO


def limpiar_lineas(linea_str: str) -> str:
    """Limpia el formato de líneas de metro.

    Reglas:
    - Convierte a string si es necesario
    - Elimina espacios en blanco
    - Extrae solo el número (elimina letras A, B, etc.)
    - Devuelve string del número

    Ejemplos:
        "10A" → "10"
        "1" → "1"
        "9B" → "9"
    """
    if pd.isna(linea_str):
        return ""

    linea_str = str(linea_str).strip()
    # Extrae solo los dígitos del inicio de la línea
    numero = ""
    for char in linea_str:
        if char.isdigit():
            numero += char
        else:
            break

    return numero


def agrupar_estaciones_metro(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa estaciones de metro por nombre normalizado y combina sus líneas.

    Pipeline:
    1. Normaliza la columna nombre (strip + title case)
    2. Extrae número de línea de cada fila (elimina letras)
    3. Agrupa por nombre
    4. Para cada grupo:
       - Mantiene lat, lon del primer registro
       - Combina líneas únicas, ordenadas, separadas por comas

    Args:
        df: DataFrame con columnas nombre, lin, lat, lon

    Returns:
        DataFrame limpio sin duplicados, con líneas agregadas
    """
    df = df.copy()

    # Normalizar nombre: eliminar espacios y estandarizar capitalización
    df["nombre"] = df["nombre"].str.strip().str.title()

    # Limpiar líneas: convertir formato de línea
    df["lin_limpia"] = df["lin"].apply(limpiar_lineas)

    # Convertir cada valor de línea en una lista para facilitar la agregación
    df["lin_list"] = (
        df["lin_limpia"]
        .where(df["lin_limpia"] != "", pd.NA)
        .apply(lambda v: [v] if pd.notna(v) else [])
    )

    # Agrupar por nombre normalizado
    grouped = (
        df.groupby("nombre", as_index=False)
        .agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            lin_list=("lin_list", lambda x: sum(x, [])),
        )
    )

    # Combinar las listas en cadenas únicas, ordenadas numéricamente
    grouped["lin"] = (
        grouped["lin_list"]
        .apply(lambda lst: sorted(set(lst), key=lambda x: int(x)))
        .apply(lambda lst: ",".join(lst))
    )

    # Seleccionar columnas finales en orden
    result = grouped[["nombre", "lat", "lon", "lin"]]

    print(f"  METRO: {len(df)} registros iniciales → {len(result)} estaciones únicas")
    return result


def limpiar_transporte(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """Selecciona y renombra las columnas de interés del dataset de transporte.

    Para BUS:
        Columnas resultantes: lin, lat, lon

    Para METRO:
        Columnas resultantes: nombre, lin, lat, lon
        Aplica deduplicación y limpieza de líneas.

    - DENOMINACION → nombre (nombre de la estación de metro / parada de bus)
    - LINEAS  → lin  (líneas que pasan por la parada / estación)
    - Latitud → lat  (Y en bus, geometry.y en metro)
    - Longitud → lon (X en bus, geometry.x en metro)
    """
    # Si el dataset tiene columnas X / Y explícitas (bus), usarlas directamente
    if "X" in df.columns and "Y" in df.columns:
        out = df[["LINEAS", "Y", "X"]].copy()
        out.columns = ["lin", "lat", "lon"]
        print(f"  {nombre}: {len(out)} registros → columnas {list(out.columns)}")
        return out

    # Si solo tiene geometría (metro), extraer coordenadas de la columna geometry
    elif "geometry" in df.columns:
        out = pd.DataFrame({
            "nombre": df["DENOMINACION"],
            "lin": df["LINEAS"],
            "lat": df["geometry"].apply(lambda geom: wkb.loads(geom).y),
            "lon": df["geometry"].apply(lambda geom: wkb.loads(geom).x),
        })

        # Aplicar deduplicación y limpieza solo a metro
        if nombre == "METRO":
            out = agrupar_estaciones_metro(out)

        print(f"  {nombre}: {len(out)} registros → columnas {list(out.columns)}")
        return out

    else:
        raise ValueError(f"No se encontraron columnas de coordenadas en {nombre}")


if __name__ == "__main__":
    cliente = crear_cliente_minio()

    for nombre, obj in {"BUS": OBJ_BUS, "METRO": OBJ_METRO}.items():
        df = bajar_minio(client=cliente, path=MINIO_RAW_SECUNDARIOS, minio_object=obj)
        df = limpiar_transporte(df, nombre)
        subir_minio(df=df, client=cliente, path=MINIO_CLEANED_SECUNDARIOS, minio_object=obj)
        print(f"OK: {nombre}")
