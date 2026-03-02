"""
test_limpieza_metro.py

Script de demostración y prueba para la limpieza de datos de metros.

Crea un dataset de ejemplo con:
- Estaciones duplicadas (mismas lat/lon)
- Líneas con formatos variados (1, 10A, 10a, 9B, etc.)

Aplica las funciones de limpieza y muestra el resultado.
"""

import pandas as pd


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

    # Convertir línea a lista vacía/uno para sumar después
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

    # Fusionar listas en cadena ordenada y única
    grouped["lin"] = (
        grouped["lin_list"]
        .apply(lambda lst: sorted(set(lst), key=lambda x: int(x)))
        .apply(lambda lst: ",".join(lst))
    )

    result = grouped[["nombre", "lat", "lon", "lin"]]

    print(f"  METRO: {len(df)} registros iniciales → {len(result)} estaciones únicas")
    return result


def crear_dataset_ejemplo() -> pd.DataFrame:
    """Crea un dataset de ejemplo con estaciones de metro duplicadas,
    líneas con letras y variaciones de nombre (espacios, capitalización)."""
    data = {
        "nombre": [
            "Sol",              # Línea 1
            "Sol ",             # Línea 2 — espacio al final
            "sol",              # Línea 3 — minúsculas
            "Gran Vía",         # Línea 1
            "Gran Vía",         # Línea 5
            "Atocha",           # Línea 1
            "ATOCHA",           # Línea a (formato inválido + mayúsculas)
            "Callao",           # Línea 3
            "Callao",           # Línea 4
            " Callao ",         # Línea 5 — espacios alrededor
            "Moncloa",          # Línea 6
            "Moncloa",          # Línea 6a (sera convertida a "6")
        ],
        "lin": [
            "1",
            "2",
            "3",
            "1",
            "5",
            "1",
            "a",  # formato inválido
            "3",
            "4",
            "5",
            "6",
            "6a",
        ],
        "lat": [
            40.416665,
            40.416665,
            40.416665,
            40.419452,
            40.419452,
            40.408840,
            40.408840,
            40.420440,
            40.420440,
            40.420440,
            40.454020,
            40.454020,
        ],
        "lon": [
            -3.603607,
            -3.603607,
            -3.603607,
            -3.608219,
            -3.608219,
            -3.691447,
            -3.691447,
            -3.616607,
            -3.616607,
            -3.616607,
            -3.718185,
            -3.718185,
        ],
    }
    return pd.DataFrame(data)


def main():
    # Crear dataset de ejemplo
    print("=" * 70)
    print("DEMOSTRACIÓN: LIMPIEZA DE ESTACIONES DE METRO")
    print("=" * 70)

    df_bruto = crear_dataset_ejemplo()
    print("\n📥 DATASET ORIGINAL (con duplicados y líneas con letras):\n")
    print(df_bruto.to_string(index=False))
    print(f"\nTotal de registros: {len(df_bruto)}")

    # Mostrar ejemplos de limpieza de líneas
    print("\n" + "=" * 70)
    print("1️⃣  LIMPIEZA DE LÍNEAS (conversión de formato)")
    print("=" * 70)
    lineas_ejemplo = ["1", "10A", "10a", "9B", "6a", "a", ""]
    for lin in lineas_ejemplo:
        limpia = limpiar_lineas(lin)
        print(f"  '{lin}' → '{limpia}'")

    # Aplicar agrupación
    print("\n" + "=" * 70)
    print("2️⃣  AGRUPACIÓN DE ESTACIONES (spatial deduplication)")
    print("=" * 70)
    df_limpiado = agrupar_estaciones_metro(df_bruto)

    print("\n📤 DATASET LIMPIO Y AGREGADO:\n")
    print(df_limpiado.to_string(index=False))
    print(f"\nTotal de registros: {len(df_limpiado)}")

    # Mostrar el resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE CAMBIOS")
    print("=" * 70)
    print(f"Registros iniciales:  {len(df_bruto)}")
    print(f"Registros finales:    {len(df_limpiado)}")
    print(f"Estaciones eliminadas: {len(df_bruto) - len(df_limpiado)}")
    print("\nDetalle de estaciones agrupadas:")
    for _, row in df_limpiado.iterrows():
        num_lineas = len(row["lin"].split(",")) if row["lin"] else 0
        print(f"  {row['nombre']:20} → {num_lineas} línea(s): {row['lin']}")

    print("\n✅ Limpieza completada correctamente")


if __name__ == "__main__":
    main()
