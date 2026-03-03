import pandas as pd
from shapely import wkb
from src.utils.funciones_minio import crear_cliente_minio, bajar_minio, subir_minio
from src.config import MINIO_PROCESSED_SECUNDARIOS, MINIO_PROCESSED_TRANSPORTE, MINIO_RAW_SECUNDARIOS, OBJ_BUS, OBJ_METRO


def limpiar_lineas(linea_str: str) -> str:
    """Limpia el formato de líneas de metro.

    Reglas:
    - Convierte a string si es necesario
    - Elimina espacios en blanco
    - Extrae solo la parte numérica (elimina letras A, B, etc.)
    - Excepción: la línea "R" (Ramal) se preserva tal cual
    - Devuelve string del número o "R"

    Ejemplos:
        "10A" → "10"    "9B" → "9"
        "1"   → "1"     "R"  → "R"
        "6a"  → "6"     "a"  → ""
    """
    if pd.isna(linea_str):
        return ""

    linea_str = str(linea_str).strip().upper()

    # Excepción: Línea R (Ramal)
    if linea_str == "R":
        return "R"

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

    # Eliminar filas sin nombre (no puede haber nulos en nombre)
    df = df.dropna(subset=["nombre"])

    # Normalizar nombre: eliminar espacios y estandarizar capitalización
    df["nombre"] = df["nombre"].astype(str).str.strip().str.upper()
    df = df[df["nombre"] != ""]
    # Limpiar líneas: convertir formato de línea
    df["linea"] = df["lin"].apply(limpiar_lineas)

    df_estaciones = df.groupby("nombre").agg({
        "linea": lambda x: list(set(x)),
        "lat":"mean",
        "lon":"mean"
    }).reset_index()
    
    # Renombramos la columna para que quede claro que ahora es una lista
    df_estaciones = df_estaciones.rename(columns={"linea": "lineas"})
    # Seleccionar columnas finales en orden
    result = df_estaciones[["nombre", "lat", "lon", "lineas"]]

    print(f"  METRO: {len(df)} registros iniciales → {len(result)} estaciones únicas")
    return result


def limpiar_transporte(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """Selecciona y renombra las columnas de interés del dataset de transporte.

    Para BUS:
        Columnas resultantes: lin, lat, lon

    Para METRO:
        Columnas resultantes: nombre, lin, lat, lon
        Aplica normalización, deduplicación y limpieza de líneas.

    - DENOMINACION → nombre (nombre de la estación de metro / parada de bus)
    - LINEAS  → lin  (líneas que pasan por la parada / estación)
    - Latitud → lat  (Y en bus, geometry.y en metro)
    - Longitud → lon (X en bus, geometry.x en metro)
    """
    if nombre == "BUS":
        out = pd.DataFrame({
            "nombre": df["DENOMINACION"],
            "lat": df["geometry"].apply(lambda geom: wkb.loads(geom).y),
            "lon": df["geometry"].apply(lambda geom: wkb.loads(geom).x),
            "lineas": df["LINEAS"],
        })
        out["lineas"] = out["lineas"].str.split(",")
        out
        print(f"  {nombre}: {len(out)} registros → columnas {list(out.columns)}")
        return out

    elif nombre == "METRO":
        out = pd.DataFrame({
            "nombre": df["DENOMINACION"],
            "lin": df["LINEAS"],
            "lat": df["geometry"].apply(lambda geom: wkb.loads(geom).y),
            "lon": df["geometry"].apply(lambda geom: wkb.loads(geom).x),
        })
        out = agrupar_estaciones_metro(out)
        print(f"  {nombre}: {len(out)} estaciones únicas → columnas {list(out.columns)}")
        return out

    else:
        raise ValueError(f"Tipo de transporte desconocido: {nombre}")


if __name__ == "__main__":
    cliente = crear_cliente_minio()

    for nombre, obj in {"BUS": OBJ_BUS, "METRO": OBJ_METRO}.items():
        df = bajar_minio(client=cliente, path=MINIO_RAW_SECUNDARIOS, minio_object=obj)
        df = limpiar_transporte(df, nombre)
        subir_minio(df=df, client=cliente, path=MINIO_PROCESSED_TRANSPORTE, minio_object=obj)
        print(f"OK: {nombre}")