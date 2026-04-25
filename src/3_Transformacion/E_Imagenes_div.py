import io
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from minio import Minio
from PIL import Image, ImageOps
import random
from utils.funciones_minio import crear_cliente_minio, bajar_minio,bajar_mapa_minio,buscar_todos_los_archivos,subir_minio
from utils.config import PATH_PRIMARIOS_LIMPIO,MODOS
LIMITE_BYTES = 350 * 1024 * 1024

def reorganizar_imagenes_por_clase(cliente:Minio, prefijo_destino="cleaned/dataset_vision"):
    
    clases = ['Cocina', 'Dormitorio', 'Salón', 'Banyo']
    
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
        nombre_path = f"cleaned/datos_primarios/imagenes/{modo}"
        objetos_origen = buscar_todos_los_archivos(cliente,nombre_path)
    
        for obj in tqdm(objetos_origen,desc = f"Dividiendo y transfiriendo imágenes de anuncios de {modo}"):
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


def reorganizar_imagenes_train_test(cliente, prefijo_destino="dataset_ml/dataset_vision", test_size=0.2):
    
    clases = ['Cocina', 'Dormitorio', 'Salón', 'Banyo']
    splits = ['train', 'test']
    
    asignacion_ids = {}
    
    buffers = {split: {clase: [] for clase in clases} for split in splits}
    tamaño_buffers = {split: {clase: 0 for clase in clases} for split in splits}
    contadores_chunks = {split: {clase: 1 for clase in clases} for split in splits}

    def vaciar_y_subir_buffer(split_nombre, clase_nombre):        
        if not buffers[split_nombre][clase_nombre]:
            return

        random.shuffle(buffers[split_nombre][clase_nombre])
        
        df_chunk = pd.DataFrame(buffers[split_nombre][clase_nombre])
        nombre_archivo = f"batch_{contadores_chunks[split_nombre][clase_nombre]}.parquet"
        
        path_archivo = f"{prefijo_destino}/{split_nombre}/{clase_nombre}"
        
        subir_minio(df_chunk, cliente, path_archivo, nombre_archivo)

        buffers[split_nombre][clase_nombre].clear()
        tamaño_buffers[split_nombre][clase_nombre] = 0
        contadores_chunks[split_nombre][clase_nombre] += 1
        
    for modo in MODOS:
        nombre_path = f"cleaned/datos_primarios/imagenes/{modo}"
        objetos_origen = buscar_todos_los_archivos(cliente, nombre_path)
    
        for obj in tqdm(objetos_origen, desc=f"Dividiendo (Train/Test) imágenes de {modo}"):
            df_original = bajar_minio(cliente, nombre_path, obj)
            
            for _, fila in df_original.iterrows():
                id_piso = fila['id']
                
                if id_piso not in asignacion_ids:
                    if random.random() < test_size:
                        asignacion_ids[id_piso] = 'test'
                    else:
                        asignacion_ids[id_piso] = 'train'
                
                split_destino = asignacion_ids[id_piso]
                
                for clase in clases:
                    if clase in df_original.columns:
                        lista_imgs = fila[clase]
                        for img_bytes in lista_imgs:
                            buffers[split_destino][clase].append({
                                'id': id_piso,
                                 'imagen_bytes': img_bytes
                            })
                            tamaño_buffers[split_destino][clase] += len(img_bytes)
                                
                            if tamaño_buffers[split_destino][clase] >= LIMITE_BYTES:
                                vaciar_y_subir_buffer(split_destino, clase)

    for split in splits:
        for clase in clases:
            vaciar_y_subir_buffer(split, clase)

def obtener_tamano_imagen(imagen_pil):
    """
    Recibe un objeto Image de Pillow y extrae sus dimensiones.
    Devuelve una tupla con (ancho, alto).
    """
    ancho, alto = imagen_pil.size
    
    print(f" Dimensiones de la imagen:")
    print(f"   - Ancho: {ancho} píxeles")
    print(f"   - Alto:  {alto} píxeles")
    
    return ancho, alto

    
if __name__ == "__main__":
    cliente = crear_cliente_minio()
    reorganizar_imagenes_por_clase(cliente)
    reorganizar_imagenes_train_test(cliente)