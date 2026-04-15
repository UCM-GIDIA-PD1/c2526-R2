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
TEMPERATURA_OLLAMA = 0.0    # Temperatura 0.0 para máxima consistencia y evitar alucinaciones

MAPEO_CLASES_EN = {
    'Cocina': 'Kitchen',
    'Dormitorio': 'Bedroom',
    'Salón': 'Living room',
    'Banyo': 'Bathroom'
}
CLASES_EN = list(MAPEO_CLASES_EN.values())

def descargar_imagenes_ejemplo(clases, max_por_clase=MAX_IMAGENES_POR_CLASE):
    """
    Descarga imágenes de ejemplo de MinIO para cada clase.
    """
    cliente = crear_cliente_minio()
    imagenes = {}
    
    for clase in clases:
        print(f"Descargando imágenes para {clase}...")
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
                        img.thumbnail((512, 512), Image.LANCZOS)
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
            # Nos aseguramos de no coger más imágenes de las que hay
            for i in range(min(num_ejemplos, len(imagenes_por_clase[clase]))):
                img_ej = imagenes_por_clase[clase][i]
                img_buffer = io.BytesIO()
                img_ej.save(img_buffer, format='JPEG')
                ejemplos.append({'clase': clase, 'bytes': img_buffer.getvalue()})
    return ejemplos

def clasificar_test_a(img_bytes, modelo, idioma):
    """
    Clasifica una imagen sin ejemplos (Zero-Shot) usando Ollama.
    """
    if idioma == 'es':
        prompt = f"""Actúa como un clasificador visual automatizado. 
Tu única tarea es clasificar la imagen adjunta dentro de una de estas clases: {CLASES_IMAGENES}.

REGLAS ESTRICTAS:
- Responde ÚNICAMENTE con la palabra exacta de la clase en español.
- NO añadas explicaciones, ni contexto, ni signos de puntuación, ni saludos.

EJEMPLO DE RESPUESTA ESPERADA:
Dormitorio
"""
    else:
        prompt = f"""Act as an automated visual classifier. 
Your only task is to classify the attached image into one of these classes: {CLASES_EN}.

STRICT RULES:
- Reply ONLY with the exact word of the class in English.
- DO NOT add explanations, context, punctuation marks, or greetings.

EXAMPLE OF EXPECTED OUTPUT:
Bedroom
"""

    images = [img_bytes]
    
    start_time = time.time()
    respuesta = ollama.generate(
        model=modelo,
        prompt=prompt,
        images=images,
        options={
            'temperature': TEMPERATURA_OLLAMA,
            'num_predict': 5 # CORRECCIÓN: Evita alucinaciones forzando respuestas muy cortas
        }
    )
    end_time = time.time()
    
    return respuesta['response'], end_time - start_time

def clasificar_test_b(img_bytes, ejemplos, modelo, idioma):
    """
    Clasifica una imagen con contextual learning (Few-Shot) usando ejemplos.
    """
    todas_las_imagenes = []
    
    if idioma == 'es':
        prompt_con_contexto = "Aquí tienes algunos ejemplos de referencia:\n"
        for i, ej in enumerate(ejemplos, 1):
            prompt_con_contexto += f"- Ejemplo {i} (Clase: {ej['clase']})\n"
            todas_las_imagenes.append(ej['bytes'])
            
        prompt_con_contexto += f"""
Actúa como un clasificador visual automatizado. 
Tu única tarea es clasificar la última imagen adjunta dentro de una de estas clases: {CLASES_IMAGENES}.

REGLAS ESTRICTAS:
- Responde ÚNICAMENTE con la palabra exacta de la clase en español.
- NO añadas explicaciones, ni contexto, ni signos de puntuación, ni saludos.

EJEMPLO DE RESPUESTA ESPERADA:
Cocina
"""
    else:
        prompt_con_contexto = "Here are some reference examples:\n"
        for i, ej in enumerate(ejemplos, 1):
            clase_ingles = MAPEO_CLASES_EN[ej['clase']]
            prompt_con_contexto += f"- Example {i} (Class: {clase_ingles})\n"
            todas_las_imagenes.append(ej['bytes'])
            
        prompt_con_contexto += f"""
Act as an automated visual classifier. 
Your only task is to classify the last attached image into one of these classes: {CLASES_EN}.

STRICT RULES:
- Reply ONLY with the exact word of the class in English.
- DO NOT add explanations, context, punctuation marks, or greetings.

EXAMPLE OF EXPECTED OUTPUT:
Kitchen
"""

    todas_las_imagenes.append(img_bytes)
    prompt = prompt_con_contexto
    images = todas_las_imagenes
    
    start_time = time.time()
    respuesta = ollama.generate(
        model=modelo,
        prompt=prompt,
        images=images,
        options={
            'temperature': TEMPERATURA_OLLAMA,
            'num_predict': 5 # CORRECCIÓN: Evita alucinaciones forzando respuestas muy cortas
        }
    )
    end_time = time.time()
    
    return respuesta['response'], end_time - start_time

def clasificar_imagen(img_bytes, test_type, modelo, idioma, ejemplos=None):
    if test_type == 'a':
        return clasificar_test_a(img_bytes, modelo, idioma)
    elif test_type == 'b':
        return clasificar_test_b(img_bytes, ejemplos, modelo, idioma)
    else:
        raise ValueError("Tipo de test desconocido.")

def extraer_clase(respuesta, clases):
    """Extrae la clase de la respuesta basándose en si contiene el nombre de la clase"""
    respuesta_lower = respuesta.lower()
    for c in clases:
        if c.lower() in respuesta_lower:
            return c
    primera_palabra = respuesta.replace('\n', ' ').strip().split(' ')
    if primera_palabra and primera_palabra[0] != '':
        return primera_palabra[0][:20]
    return "Unknown"

def main():
    print("¡Bienvenido al comparador de Zero-Shot vs Few-Shot!")
    
    print("\n--- MENÚ DE CONFIGURACIÓN ---")
    print("Selecciona el modelo a utilizar:")
    print("1) llava")
    print("2) bakllava")
    opcion_modelo = input("Opción (1/2) [por defecto 1]: ").strip()
    modelo_seleccionado = 'bakllava' if opcion_modelo == '2' else 'llava'
    
    print("\nSelecciona el idioma del prompt:")
    print("1) Español")
    print("2) Inglés")
    opcion_idioma = input("Opción (1/2) [por defecto 2]: ").strip()
    idioma = 'es' if opcion_idioma == '1' else 'en'
    
    print(f"\nConfiguración final -> Modelo: {modelo_seleccionado} | Idioma: {idioma}")
    
    # Inicializar wandb
    wandb.init(
        project="proyecto-vlm-minio",
        name=f"test-zero-vs-few-shot-{modelo_seleccionado}-{idioma}",
        config={
            "modelo": modelo_seleccionado,
            "idioma": idioma,
            "temperatura": TEMPERATURA_OLLAMA,
            "max_imagenes_por_clase": MAX_IMAGENES_POR_CLASE,
            "ejemplos_por_clase": NUM_EJEMPLOS_CONTEXTO
        }
    )
    
    # Crear la tabla de wandb
    columnas = ["imagen", "clase_real", "pred_clase_a", "time_a", "acierto_a", "pred_clase_b", "time_b", "acierto_b"]
    tabla_wandb = wandb.Table(columns=columnas)
    
    # Descargar imágenes
    print("\nDescargando imágenes de MinIO...")
    imagenes_por_clase = descargar_imagenes_ejemplo(CLASES_IMAGENES)
    
    # Preparar ejemplos
    print("\nPreparando ejemplos para contextual learning (Test B)...")
    ejemplos = preparar_ejemplos(imagenes_por_clase)
    print(f"Se usarán {len(ejemplos)} ejemplos en total.")
    
    print("\nClasificando imágenes (Comparando Test A vs Test B)...")
    
    # Contador global para las gráficas de W&B
    paso_evaluacion = 1
    
    for clase_real, lista_imagenes in imagenes_por_clase.items():
        print(f"\n--- Evaluando clase: {clase_real} ---")
        
        # CORRECCIÓN CRÍTICA (Data Leakage): 
        # Omitimos las primeras N imágenes que ya usamos como ejemplos en 'preparar_ejemplos'
        imagenes_para_test = lista_imagenes[NUM_EJEMPLOS_CONTEXTO:]
        
        # Si no quedan imágenes para testear en esta clase, la saltamos
        if not imagenes_para_test:
            print(f"No hay suficientes imágenes para testear la clase {clase_real} después de extraer ejemplos.")
            continue
            
        for i, img in enumerate(imagenes_para_test, start=NUM_EJEMPLOS_CONTEXTO + 1):
            
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG')
            img_bytes = img_buffer.getvalue()
            
            clases_evaluacion = CLASES_IMAGENES if idioma == 'es' else CLASES_EN
            clase_real_evaluacion = clase_real if idioma == 'es' else MAPEO_CLASES_EN[clase_real]
            
            # --- TEST A ---
            print(f"\n[Imagen {i}/{len(lista_imagenes)}] Ejecutando Test A (Zero-shot)...")
            respuesta_a, time_a = clasificar_test_a(img_bytes, modelo_seleccionado, idioma)
            pred_clase_a = extraer_clase(respuesta_a, clases_evaluacion)
            acierto_a = clase_real_evaluacion.lower() in respuesta_a.lower()
            print(f"  -> Predicción: {pred_clase_a} | Acierto: {acierto_a} | Tiempo: {time_a:.2f}s")
            
            # --- TEST B ---
            print(f"[Imagen {i}/{len(lista_imagenes)}] Ejecutando Test B (Few-shot)...")
            respuesta_b, time_b = clasificar_test_b(img_bytes, ejemplos, modelo_seleccionado, idioma)
            pred_clase_b = extraer_clase(respuesta_b, clases_evaluacion)
            acierto_b = clase_real_evaluacion.lower() in respuesta_b.lower()
            print(f"  -> Predicción: {pred_clase_b} | Acierto: {acierto_b} | Tiempo: {time_b:.2f}s")
            
            # CORRECCIÓN: Registrar métricas paso a paso en W&B para generar gráficas de líneas
            wandb.log({
                "evaluacion_paso": paso_evaluacion,
                "tiempo_inferencia/Test_A_segundos": time_a,
                "tiempo_inferencia/Test_B_segundos": time_b,
                "precision_acumulada/Acierto_A": int(acierto_a), # 1 o 0
                "precision_acumulada/Acierto_B": int(acierto_b)
            })
            paso_evaluacion += 1
            
            # Añadir fila a la tabla visual
            tabla_wandb.add_data(
                wandb.Image(img), 
                clase_real_evaluacion, 
                pred_clase_a, 
                round(time_a, 2), 
                bool(acierto_a), 
                pred_clase_b, 
                round(time_b, 2), 
                bool(acierto_b)
            )
            
    print("\nSubiendo resultados finales a Weights & Biases...")
    wandb.log({"Tabla_Analisis_Visual": tabla_wandb})
    wandb.finish()
    
    print("\n✅ ¡Experimento finalizado exitosamente! Revisa tu dashboard en wandb.ai")

if __name__ == "__main__":
    main()