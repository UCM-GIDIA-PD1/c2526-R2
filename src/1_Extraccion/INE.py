import os
import requests
import pandas as pd
from pathlib import Path
from minio import Minio
from dotenv import load_dotenv

ruta_original = Path(__file__).resolve().parent
ruta_datos = ruta_original / 'datos_secundarios' / 'ine_renta'
ruta_datos.mkdir(parents=True, exist_ok=True)

url_ine = "https://servicios.ine.es/wstempus/js/es/DATOS_TABLA/30824?tip=AM"
nombre_parquet = ruta_datos / "renta_hogar_secciones_madrid.parquet"

def descargar_y_procesar():
    response = requests.get(url_ine)
    
    if response.status_code == 200:
        print("Descargando datos del INE...")
        df = pd.DataFrame(response.json())

        def sacar_codigo_seccion(metadata_lista):
            try:
                for item in metadata_lista:
                    # El código de sección tiene 10 dígitos, por ejemplo: 2807901001
                    # 28 Madrid, 079 Madrid Capital, 01 Distrito (Centro), 001 Sección
                    if 'Codigo' in item and len(item['Codigo']) == 10: 
                        return item['Codigo']
            except:
                pass
            return None

        df['CUSEC'] = df['MetaData'].apply(sacar_codigo_seccion)

        df_madrid = df[df['CUSEC'].str.startswith('28079', na=False)].copy()
        
        # Buscamos solo las filas de Renta neta media por hogar
        filtro_renta = df_madrid['Nombre'].str.contains('Renta neta media por hogar', case=False, na=False)
        df_madrid = df_madrid[filtro_renta]

        # Convertimos cada elemento (año) de la lista 'Data' en una fila separada
        df_exploded = df_madrid.explode('Data')
        datos_extraidos = df_exploded['Data'].apply(pd.Series)
        df_final = pd.concat([df_exploded[['CUSEC']], datos_extraidos], axis=1)
        
        # Filtramos por el último año disponible
        df_final['Anyo'] = pd.to_numeric(df_final['Anyo'], errors='coerce')
        ultimo_ano = df_final['Anyo'].max()
        df_final = df_final[df_final['Anyo'] == ultimo_ano]

        df_final.rename(columns={'Valor': 'renta_media'}, inplace=True)
        df_exportar = df_final[['CUSEC', 'renta_media']].dropna().drop_duplicates()

        # Guardamos en Parquet
        df_exportar.to_parquet(nombre_parquet)
        
        return nombre_parquet
                
    else:
        print(f"ERROR: {response.status_code}")
        return None

def subir_a_minio(ruta_archivo):
    if not ruta_archivo: return
    
    print("Subiendo archivo a MinIO...")
    load_dotenv()
    minio_bucket = os.getenv("MINIO_BUCKET")
    minio_group = os.getenv("GROUP_PATH")
    path_subcarpeta = "datos_secundarios/ine"

    client = Minio(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
    )

    nombre_objeto = f"{minio_group}/{path_subcarpeta}/{ruta_archivo.name}"
    
    client.fput_object(
        bucket_name=minio_bucket,
        object_name=nombre_objeto,
        file_path=str(ruta_archivo)
    )
    
    # Limpiamos los archivos locales
    ruta_archivo.unlink() # Borra el parquet
    if ruta_datos.exists():
        # Borra la carpeta temporal si está vacía
        try:
            ruta_datos.rmdir()
            (ruta_original / 'datos_secundarios').rmdir()
        except:
            pass 

if __name__ == "__main__":
    print("Iniciando el proceso de extracción del INE... (Tarda un par de minutos)")

    archivo = descargar_y_procesar()
    print("Archivo descargado y convertido a parquet")

    subir_a_minio(archivo)
    print("Archivo subido a MinIO")
