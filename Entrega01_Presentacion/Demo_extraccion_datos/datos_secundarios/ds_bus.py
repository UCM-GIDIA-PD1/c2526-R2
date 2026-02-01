import pandas as pd

# 1. Cargar el archivo de autobuses
archivo_bus = "M6_Estaciones.csv"

try:
    # Usamos sep=None para que detecte si es coma (,) o punto y coma (;) automáticamente
    df_bus = pd.read_csv(archivo_bus, sep=None, engine='python', encoding='utf-8')

    # 2. Selección de las columnas solicitadas
    columnas_bus = ['DENOMINACIONABREVIADA', 'X', 'Y', 'LINEAS']

    # Verificamos que las columnas existan (por si acaso hay variaciones en el nombre)
    columnas_reales = [c for c in columnas_bus if c in df_bus.columns]
    df_bus_filtrado = df_bus[columnas_reales].copy()

    # 3. Limpieza profunda (Estrategia para evitar duplicados como el de Puerta del Sur)
    # Eliminamos nulos en el nombre
    df_bus_filtrado = df_bus_filtrado.dropna(subset=['DENOMINACIONABREVIADA'])

    # Normalizamos el texto: quitamos espacios y ponemos formato Título
    df_bus_filtrado['DENOMINACIONABREVIADA'] = (
        df_bus_filtrado['DENOMINACIONABREVIADA']
        .astype(str)
        .str.strip()
        .str.title()
    )

    # 4. Eliminamos duplicados
    # En buses es normal que una parada tenga varias líneas; si solo quieres la ubicación de la parada:
    df_bus_final = df_bus_filtrado.drop_duplicates(subset=['DENOMINACIONABREVIADA', 'X', 'Y'])

    # 5. Guardar el resultado limpio
    df_bus_final.to_csv('Bus_Madrid_Limpio.csv', index=False, encoding='utf-8-sig')

    print(f"¡Éxito! Dataset de autobuses creado con {len(df_bus_final)} paradas.")
    print(df_bus_final.head())

except Exception as e:
    print(f"Error al procesar el archivo de bus: {e}")