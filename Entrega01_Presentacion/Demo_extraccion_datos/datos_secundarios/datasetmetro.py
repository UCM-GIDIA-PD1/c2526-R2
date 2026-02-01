import pandas as pd
import glob

# 1. Buscamos los archivos
archivos = glob.glob("M4_L*_S1_ESTACION.csv")
lista_dataframes = []

for archivo in archivos:
    try:
        # Cargamos el archivo (autodetectando el separador)
        df_temp = pd.read_csv(archivo, sep=None, engine='python', encoding='utf-8')

        # Normalizamos nombres de columnas a mayúsculas
        df_temp.columns = [c.upper() for c in df_temp.columns]
        if 'DENOMICI' in df_temp.columns:
            df_temp = df_temp.rename(columns={'DENOMICI': 'DENOMINACION'})

        # --- MEJORA DE LIMPIEZA ---
        # 1. Eliminamos filas donde la denominación sea totalmente nula
        df_temp = df_temp.dropna(subset=['DENOMINACION'])

        # 2. Limpiamos espacios en blanco y normalizamos el texto
        df_temp['DENOMINACION'] = df_temp['DENOMINACION'].astype(str).str.strip().str.title()

        # Seleccionamos columnas
        columnas_interes = ['DENOMINACION', 'X', 'Y', 'DIRECCION', 'LINEAS']
        columnas_presentes = [col for col in columnas_interes if col in df_temp.columns]

        lista_dataframes.append(df_temp[columnas_presentes])

    except Exception as e:
        print(f"Error en {archivo}: {e}")

# 2. Unión y limpieza final de duplicados
if lista_dataframes:
    df_final = pd.concat(lista_dataframes, ignore_index=True)

    # Eliminamos duplicados basándonos solo en el nombre limpio
    # 'first' mantiene la primera aparición (puedes cambiar a 'last' si prefieres)
    df_final = df_final.drop_duplicates(subset=['DENOMINACION'], keep='first')

    # 3. Guardar el resultado
    df_final.to_csv('Metro_Madrid_Limpio.csv', index=False, encoding='utf-8-sig')
    print(f"¡Hecho! 'Puerta del Sur' ahora debería aparecer una sola vez y sin nulos.")