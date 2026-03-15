import io
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from minio import Minio
from utils.funciones_minio import crear_cliente_minio, bajar_minio,bajar_mapa_minio,buscar_todos_los_archivos,subir_minio
from utils.config import PATH_PRIMARIOS_LIMPIO,MODOS
LIMITE_BYTES = 350 * 1024 * 1024

def reorganizar_imagenes_por_clase(cliente:Minio, prefijo_destino="dataset_vision"):
    
    cliente = crear_cliente_minio()
    clases = ['Cocina', 'Dormitorio', 'Salón', 'Banyo', 'Comedor']
    
    buffers = {clase: [] for clase in clases}
    tamaño_buffers = {clase: 0 for clase in clases}
    contadores_chunks = {clase: 1 for clase in clases}
    
 

    def vaciar_y_subir_buffer(clase_nombre):        
        df_chunk = pd.DataFrame(buffers[clase_nombre])
        nombre_archivo = f"batch_{contadores_chunks[clase_nombre]}.parquet"
        path_archivo = f"{prefijo_destino}/{clase_nombre}"
        subir_minio(df_chunk,cliente,path_archivo,nombre_archivo)

        buffers[clase_nombre].clear()
        tamaño_buffers[clase_nombre] = 0
        contadores_chunks[clase_nombre]+=1
        
    for modo in MODOS:
        nombre_path = f"datos_primarios/imagenes/{modo}"
        objetos_origen = buscar_todos_los_archivos(cliente,nombre_path)
    
        for obj in tqdm(objetos_origen):
            df_original = bajar_minio(cliente,nombre_path,obj)
            for _, fila in df_original.iterrows():
                id_piso = fila['id']
                for clase in clases:
                    if clase in df_original.columns:
                        lista_imgs = fila[clase]
                        for img_bytes in lista_imgs:
                            if isinstance(img_bytes, bytes):
                                buffers[clase].append({
                                    'id': id_piso,
                                    'imagen_bytes': img_bytes
                                })
                                tamaño_buffers[clase] += len(img_bytes)
                                    
                                if tamaño_buffers[clase] >= LIMITE_BYTES:
                                    vaciar_y_subir_buffer(clase)

    for clase in clases:
        vaciar_y_subir_buffer(clase)


if __name__ == "__main__":
    cliente = crear_cliente_minio()
    reorganizar_imagenes_por_clase(cliente,)