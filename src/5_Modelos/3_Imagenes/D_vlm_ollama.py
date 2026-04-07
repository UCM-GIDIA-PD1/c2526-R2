import ollama
import time
import io
import random
import wandb
from PIL import Image
from utils.config import CLASES_IMAGENES, MINIO_DATASET_VISION
from utils.funciones_minio import crear_cliente_minio, bajar_minio, buscar_todos_los_archivos

# Configuración fácil de cambiar
MAX_IMAGENES_POR_CLASE = 5  # Cuántas imágenes descargar por clase
NUM_EJEMPLOS_CONTEXTO = 1   # Cuántos ejemplos usar por clase en Test B
TEMPERATURA_OLLAMA = 0.1    # Temperatura para el modelo (baja para consistencia) si es alta la respuesta es mas aleatoria

def descargar_imagenes_ejemplo(clases, max_por_clase=MAX_IMAGENES_POR_CLASE):
    """
    Descarga imágenes de ejemplo de MinIO para cada clase.
    Esto es para tener datos para probar el modelo.
    """
    cliente = crear_cliente_minio()
    imagenes = {}
    
    for clase in clases:
        print(f"Descargando imgenes para {clase}...")
        try:
            archivos = buscar_todos_los_archivos(cliente, f"{MINIO_DATASET_VISION}/{clase}")
            if not archivos:
                print(f"No se encontraron archivos para {clase}")
                continue
            
            # Tomar algunos archivos aleatorios
            archivos_seleccionados = random.sample(archivos, min(len(archivos), 3))
            imagenes[clase] = []
            
            for archivo in archivos_seleccionados:
                try:
                    df = bajar_minio(cliente, f"{MINIO_DATASET_VISION}/{clase}", archivo)
                    # Tomar algunas imágenes del dataframe
                    num_imagenes = min(len(df), max_por_clase // len(archivos_seleccionados) + 1)
                    for i in range(num_imagenes):
                        if len(imagenes[clase]) >= max_por_clase:
                            break
                        img_bytes = df['imagen_bytes'].iloc[i]
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        img.thumbnail((512, 512), Image.LANCZOS)  # Hacer más pequeñas para que sea rápido
                        imagenes[clase].append(img)
                    if len(imagenes[clase]) >= max_por_clase:
                        break
                except Exception as e:
                    print(f"Error descargando archivo {archivo}: {e}")
                    continue
        except Exception as e:
            print(f"Error descargando {clase}: {e}")
    
    return imagenes

def preparar_ejemplos(imagenes_por_clase, num_ejemplos=NUM_EJEMPLOS_CONTEXTO):
    """
    Prepara ejemplos para el contextual learning.
    Toma las primeras imágenes de cada clase como ejemplos.
    """
    ejemplos = []
    for clase in CLASES_IMAGENES:
        if clase in imagenes_por_clase and imagenes_por_clase[clase]:
            for i in range(min(num_ejemplos, len(imagenes_por_clase[clase]))):
                img_ej = imagenes_por_clase[clase][i]
                img_buffer = io.BytesIO()
                img_ej.save(img_buffer, format='JPEG')
                ejemplos.append({'clase': clase, 'bytes': img_buffer.getvalue()})
    return ejemplos

def clasificar_test_a(img_bytes):
    """
    Clasifica una imagen sin ejemplos usando Ollama.
    """
    prompt = f"""
    Analiza esta imagen y clasificala. Responde estrictamente en español con este formato de tres puntos y sé muy breve:
    1. **Clase:** [Selecciona una de: {CLASES_IMAGENES}]
    2. **Razón:** [Explicación muy breve]
    3. **Similitud:** [Si se parece a otra clase, di cuál es y por qué muy brevemente. Si no, di 'Ninguna']
    No añadas introducciones ni saludos.
    """
    images = [img_bytes]
    
    start_time = time.time()
    respuesta = ollama.generate(
        model='llava',
        prompt=prompt,
        images=images,
        options={'temperature': TEMPERATURA_OLLAMA}
    )
    end_time = time.time()
    
    return respuesta['response'], end_time - start_time

def clasificar_test_b(img_bytes, ejemplos):
    """
    Clasifica una imagen con contextual learning usando ejemplos.
    """
    prompt_con_contexto = "Actúa como un clasificador experto. Te voy a dar ejemplos primero:\n"
    
    todas_las_imagenes = []
    for i, ej in enumerate(ejemplos, 1):
        prompt_con_contexto += f"La imagen {i} es de la clase: {ej['clase']}\n"
        todas_las_imagenes.append(ej['bytes'])
    
    # Añadimos la imagen final
    prompt_con_contexto += f"\nAhora, clasifica la última imagen en una de estas categorías: {CLASES_IMAGENES}. Responde solo con la clase."
    todas_las_imagenes.append(img_bytes)
    prompt = prompt_con_contexto
    images = todas_las_imagenes
    
    start_time = time.time()
    respuesta = ollama.generate(
        model='llava',
        prompt=prompt,
        images=images,
        options={'temperature': TEMPERATURA_OLLAMA}
    )
    end_time = time.time()
    
    return respuesta['response'], end_time - start_time

def clasificar_imagen(img_bytes, test_type, ejemplos=None):
    """
    Función envoltorio para clasificar la imagen dependiendo del tipo de test.
    """
    if test_type == 'a':
        return clasificar_test_a(img_bytes)
    elif test_type == 'b':
        return clasificar_test_b(img_bytes, ejemplos)
    else:
        raise ValueError("Tipo de test desconocido.")

def extraer_clase(respuesta, clases):
    """Extrae la clase de la respuesta basándose en si contiene el nombre de la clase"""
    respuesta_lower = respuesta.lower()
    for c in clases:
        if c.lower() in respuesta_lower:
            return c
    # Si no la encuentra claramente, devolver los primeros 30 caracteres
    return respuesta.replace('\n', ' ').strip()[:30]

def main():
    """
    Función principal del programa.
    Descarga imágenes y las clasifica con Test A y Test B para comparar.
    Sube los resultados a Weights & Biases.
    """
    print("¡Bienvenido al comparador de Zero-Shot vs Few-Shot con LLaVA!")
    
    # Inicializar wandb
    wandb.init(
        project="proyecto-vlm-minio",
        name="test-zero-vs-few-shot",
        config={
            "modelo": "llava",
            "temperatura": TEMPERATURA_OLLAMA,
            "max_imagenes_por_clase": MAX_IMAGENES_POR_CLASE
        }
    )
    
    # Crear la tabla de wandb
    columnas = ["imagen", "clase_real", "pred_clase_a", "time_a", "acierto_a", "pred_clase_b", "time_b", "acierto_b"]
    tabla_wandb = wandb.Table(columns=columnas)
    
    # Descargar imágenes
    print("\nDescargando imágenes de MinIO...")
    imagenes_por_clase = descargar_imagenes_ejemplo(CLASES_IMAGENES)
    
    # Preparar ejemplos para contextual learning
    print("\nPreparando ejemplos para contextual learning (Test B)...")
    ejemplos = preparar_ejemplos(imagenes_por_clase)
    print(f"Se usarán {len(ejemplos)} ejemplos.")
    
    # Clasificar imágenes
    total_imagenes = sum(len(imgs) for imgs in imagenes_por_clase.values())
    print(f"\nClasificando {total_imagenes} imágenes comparando Test A vs Test B...")
    
    for clase_real, lista_imagenes in imagenes_por_clase.items():
        print(f"\n--- Clasificando imágenes de {clase_real} ---")
        for i, img in enumerate(lista_imagenes):
            # Convertir imagen a bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG')
            img_bytes = img_buffer.getvalue()
            
            # Ejecutar Test A (Zero-shot)
            print(f"\nEjecutando Test A (Zero-shot) para imagen {i+1} de {clase_real}...")
            respuesta_a, time_a = clasificar_imagen(img_bytes, 'a')
            pred_clase_a = extraer_clase(respuesta_a, CLASES_IMAGENES)
            acierto_a = clase_real.lower() in respuesta_a.lower()
            print(f"Test A -> Predicción: {pred_clase_a} | Acierto: {acierto_a} | Tiempo: {time_a:.2f}s")
            
            # Ejecutar Test B (Few-shot)
            print(f"Ejecutando Test B (Few-shot) para imagen {i+1} de {clase_real}...")
            respuesta_b, time_b = clasificar_imagen(img_bytes, 'b', ejemplos)
            pred_clase_b = extraer_clase(respuesta_b, CLASES_IMAGENES)
            acierto_b = clase_real.lower() in respuesta_b.lower()
            print(f"Test B -> Predicción: {pred_clase_b} | Acierto: {acierto_b} | Tiempo: {time_b:.2f}s")
            
            # 2. Añadir fila a la tabla visual (La magia de W&B para VLMs)
            tabla_wandb.add_data(
                wandb.Image(img), 
                clase_real, 
                pred_clase_a, 
                time_a, 
                bool(acierto_a), 
                pred_clase_b, 
                time_b, 
                bool(acierto_b)
            )
            
    print("\nRegistrando resultados en Weights & Biases...")
    wandb.log({"resultados_comparativa": tabla_wandb})
    wandb.finish()
    
    print("\n¡Clasificación y subida a W&B terminada exitosamente!")

if __name__ == "__main__":
    main()