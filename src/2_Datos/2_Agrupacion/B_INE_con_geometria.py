import io
import pandas
from src.utils.funciones_minio import crear_cliente_minio, minio_subir_memoria, bajar_minio
from src.config import MINIO_INE, MINIO_RAW_SECUNDARIOS, MINIO_REJILLAS_SUCIO, OBJ_INE, OBJ_INE_JUNTO, OBJ_SECCIONES

def unir_datos_ine_seccion_censal():
    """
    Descarga los datos de renta del INE y las geometrías de las secciones censales desde MinIO,
    los une por el código CUSEC y sube el resultado unido a MinIO en formato Parquet.
    """
    print("Iniciando proceso de unión de datos del INE con secciones censales... \n")
    client = crear_cliente_minio()
    
    print("Descargando datos del INE y secciones censales desde MinIO... \n")
    df_ine = bajar_minio(client, MINIO_RAW_SECUNDARIOS, OBJ_INE)
    df_secciones_censales = bajar_minio(client, MINIO_REJILLAS_SUCIO, OBJ_SECCIONES)

    print("Uniendo datos del INE con geometrías de secciones censales... \n")

    # Para que no queden huecos en el mapa, rellenamos las rentas nulas con la media de su distrito
    df_final = df_ine.merge(df_secciones_censales, on='CUSEC', how='right')

    df_final['DISTRITO'] = df_final['CUSEC'].str[5:7]

    df_final['renta_media'] = df_final.groupby('DISTRITO')['renta_media'].transform(
        lambda x: x.fillna(x.mean())
    )
    
    df_final = df_final.drop(columns=['DISTRITO'])
    df_final = df_final.drop_duplicates(subset=['CUSEC'], keep='first')
    
    print("Guardando datos unidos en formato Parquet y subiendo a MinIO... \n")
    buffer = io.BytesIO()
    df_final.to_parquet(buffer, index=False)
    buffer.seek(0)

    minio_subir_memoria(client, MINIO_INE, OBJ_INE_JUNTO, buffer)
    print("Datos unidos del INE con secciones censales subidos a MinIO. \n")

if __name__ == "__main__":
    unir_datos_ine_seccion_censal()