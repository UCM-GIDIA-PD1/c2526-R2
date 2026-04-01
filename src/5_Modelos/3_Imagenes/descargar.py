import os
import io
from PIL import Image
from utils.funciones_minio import crear_cliente_minio, bajar_minio, buscar_todos_los_archivos
from utils.config import CLASES_IMAGENES, CNN_TARGET_SIZE

def descargar_a_disco(clases):
    cliente = crear_cliente_minio()
    base_dir = "dataset_local"
    
    for clase in clases:
        path_clase = os.path.join(base_dir, clase)
        os.makedirs(path_clase, exist_ok=True)
        
        # Buscamos todos los parquets de esa clase
        archivos = buscar_todos_los_archivos(cliente, f"cleaned/dataset_vision/{clase}")
        print(f"Descargando {clase}...")

        for n_arc, archivo in enumerate(archivos):
            df = bajar_minio(cliente, f"cleaned/dataset_vision/{clase}", archivo)
            for i, raw in enumerate(df['imagen_bytes']):
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                # Redimensionamos aquí para que el archivo pesado ya sea pequeño
                img = img.resize((CNN_TARGET_SIZE[1], CNN_TARGET_SIZE[0]))
                img.save(os.path.join(path_clase, f"{n_arc}_{i}.jpg"))
            del df

if __name__ == "__main__":
    descargar_a_disco(CLASES_IMAGENES)