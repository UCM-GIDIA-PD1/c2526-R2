import ollama
import time
import io
import random
from PIL import Image
from utils.config import CLASES_IMAGENES, MINIO_DATASET_VISION
from utils.funciones_minio import crear_cliente_minio, bajar_minio, buscar_todos_los_archivos

# Configuración fácil de cambiar
MAX_IMAGENES_POR_CLASE = 5  # Cuántas imágenes descargar por clase
NUM_EJEMPLOS_CONTEXTO = 1   # Cuántos ejemplos usar por clase en Test B
TEMPERATURA_OLLAMA = 0.2    # Temperatura para el modelo (baja para consistencia) si es alta la respuesta es mas aleatoria

def descargar_imagenes_ejemplo(clases, max_por_clase=MAX_IMAGENES_POR_CLASE):
    """
    Descarga imágenes de ejemplo de MinIO para cada clase.
    Esto es para tener datos para probar el modelo.
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

def main():
    """
    Función principal del programa.
    Descarga imágenes, pregunta qué test hacer y clasifica.
    """
    print("Bienvenido al clasificador de imágenes con LLaVA!")
    print("Este es un proyecto de datos para clasificar habitaciones de casas.")
    
    # Descargar imágenes
    print("\nDescargando imágenes de MinIO...")
    imagenes_por_clase = descargar_imagenes_ejemplo(CLASES_IMAGENES)
    
    # Seleccionar test
    test_type = input("\nElige test A (sin ejemplos) o B (con contextual learning): ").strip().lower()
    while test_type not in ['a', 'b']:
        test_type = input("Por favor, elige 'a' o 'b': ").strip().lower()
    
    # Preparar ejemplos si es Test B
    ejemplos = []
    if test_type == 'b':
        print("Preparando ejemplos para contextual learning...")
        ejemplos = preparar_ejemplos(imagenes_por_clase)
        print(f"Se usarán {len(ejemplos)} ejemplos.")
    
    # Clasificar imágenes
    total_imagenes = sum(len(imgs) for imgs in imagenes_por_clase.values())
    print(f"\nClasificando {total_imagenes} imágenes...")
    
    for clase, lista_imagenes in imagenes_por_clase.items():
        print(f"\n--- Clasificando imágenes de {clase} ---")
        for i, img in enumerate(lista_imagenes):
            # Convertir imagen a bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG')
            img_bytes = img_buffer.getvalue()
            
            # Clasificar
            respuesta, tiempo = clasificar_imagen(img_bytes, test_type, ejemplos)
            
            print(f"Tiempo: {tiempo:.2f} segundos")
            print(f"Imagen {i+1} de {clase}:")
            print(f"  Resultado: {respuesta}")
            print("-" * 50)
    
    print("\n¡Clasificación terminada!")

if __name__ == "__main__":
    main()