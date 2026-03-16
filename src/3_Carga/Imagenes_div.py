import io
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from minio import Minio
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from utils.funciones_minio import crear_cliente_minio, bajar_minio,bajar_mapa_minio,buscar_todos_los_archivos,subir_minio
from utils.config import PATH_PRIMARIOS_LIMPIO,MODOS
LIMITE_BYTES = 350 * 1024 * 1024

class LetterboxPad:
    def __init__(self, target_size=224, color=(0, 0, 0)):
        self.target_size = (target_size, target_size)
        self.color = color

    def __call__(self, img):
        return ImageOps.pad(img, self.target_size, color=self.color)

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

def embeddings_imagenes(cliente, batch_size=32):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pesos = models.ResNet50_Weights.DEFAULT
    modelo_resnet = models.resnet50(weights=pesos)
    modelo_resnet.fc = nn.Identity() 
    modelo_resnet.eval()
    modelo_resnet.to(device)
    
    mis_transforms = transforms.Compose([
        LetterboxPad(target_size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    clases = ['Cocina', 'Dormitorio', 'Salón', 'Banyo']
    resultados_finales = []
    
    def procesar_lote(tensores, ids, etiquetas):
        if len(tensores) == 0: return

        batch_tensor = torch.stack(tensores).to(device)
        
        with torch.no_grad():
            embeddings = modelo_resnet(batch_tensor)
            
        embeddings_np = embeddings.cpu().numpy().astype(np.float16)
        
        for i in range(len(ids)):
            resultados_finales.append({
                'id': ids[i],
                'clase': etiquetas[i],
                'embedding': embeddings_np[i]
            })
    
    for clase in clases:
        path = f"dataset_vision/{clase}"
        objetos = buscar_todos_los_archivos(cliente,path)
        
        for obj in tqdm(objetos,desc = f"Procesando imagenes de {clase}"):
            
            df_chunk = bajar_minio(cliente,path,obj)
            
            lote_tensores = []
            lote_ids = []
            lote_clases = []
            
            for _, fila in df_chunk.iterrows():
                img = Image.open(io.BytesIO(fila['imagen_bytes'])).convert("RGB")
                tensor_listo = mis_transforms(img)
                    
                lote_tensores.append(tensor_listo)
                lote_ids.append(fila['id'])
                lote_clases.append(clase)
                    
                if len(lote_tensores) == batch_size:
                    procesar_lote(lote_tensores, lote_ids, lote_clases)
                    lote_tensores, lote_ids, lote_clases = [], [], []
            
            procesar_lote(lote_tensores, lote_ids, lote_clases)

    df_final = pd.DataFrame(resultados_finales)
    
    subir_minio(df_final,cliente,"dataset_ml","embeddings_imagenes.parquet")
    
if __name__ == "__main__":
    cliente = crear_cliente_minio()
    reorganizar_imagenes_por_clase(cliente)
    embeddings_imagenes(cliente)