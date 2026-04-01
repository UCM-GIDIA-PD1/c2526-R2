import os
import io
import random  # Agregado para mezclar aleatoriamente
from PIL import Image
from utils.funciones_minio import crear_cliente_minio, bajar_minio, buscar_todos_los_archivos
from utils.config import CLASES_IMAGENES, CNN_TARGET_SIZE

def descargar_a_disco(clases, max_por_clase=None):
    """
    Descarga imágenes desde MinIO y las guarda localmente.
    
    Args:
        clases (list): Lista de clases a descargar.
        max_por_clase (int, optional): Máximo número de imágenes por clase. Si None, descarga todas.
    """
    cliente = crear_cliente_minio()
    base_dir = "dataset_local"
    
    for clase in clases:
        path_clase = os.path.join(base_dir, clase)
        os.makedirs(path_clase, exist_ok=True)
        
        # Buscamos todos los parquets de esa clase
        archivos = buscar_todos_los_archivos(cliente, f"cleaned/dataset_vision/{clase}")
        print(f"Descargando {clase}... (límite: {max_por_clase if max_por_clase else 'todos'})")
        
        contador_clase = 0  # Contador para el límite por clase
        
        for n_arc, archivo in enumerate(archivos):
            if max_por_clase and contador_clase >= max_por_clase:
                break  # Detener si ya alcanzamos el límite
            
            df = bajar_minio(cliente, f"cleaned/dataset_vision/{clase}", archivo)
            
            # Mezclar aleatoriamente las filas del DataFrame para muestreo aleatorio
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            for i, raw in enumerate(df['imagen_bytes']):
                if max_por_clase and contador_clase >= max_por_clase:
                    break  # Detener si ya alcanzamos el límite
                
                try:
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    # Redimensionamos aquí para que el archivo pesado ya sea pequeño
                    img = img.resize((CNN_TARGET_SIZE[1], CNN_TARGET_SIZE[0]))
                    img.save(os.path.join(path_clase, f"{n_arc}_{i}.jpg"))
                    contador_clase += 1
                except Exception as e:
                    print(f"Error procesando imagen {n_arc}_{i} en {clase}: {e}")
                    continue  # Saltar imágenes corruptas
            
            del df
        
        print(f"Clase {clase}: {contador_clase} imágenes descargadas.")

if __name__ == "__main__":
    # Cambia 15000 por el número deseado por clase (60k total / 4 clases ≈ 15k)
    descargar_a_disco(CLASES_IMAGENES, max_por_clase=15000)
    