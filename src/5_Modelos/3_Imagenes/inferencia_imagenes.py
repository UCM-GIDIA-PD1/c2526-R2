import tensorflow as tf
import numpy as np
from PIL import Image
import os
from utils.config import CLASES_IMAGENES

# Configuración del modelo
IMAGE_SIZE = (160, 240)  # Debe coincidir con el entrenamiento

def cargar_modelo(ruta_modelo="modelo_imagenes.keras"):
    """Carga el modelo entrenado desde disco."""
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"No se encontró el modelo en {ruta_modelo}")

    modelo = tf.keras.models.load_model(ruta_modelo)
    print(f"Modelo cargado desde {ruta_modelo}")
    return modelo

def preprocesar_imagen(ruta_imagen):
    """Preprocesa una imagen para la predicción."""
    # Cargar imagen
    img = Image.open(ruta_imagen).convert('RGB')

    # Redimensionar
    img = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))  # (width, height)

    # Convertir a array y normalizar
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Añadir dimensión batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predecir_imagen(modelo, ruta_imagen):
    """Hace una predicción sobre una imagen."""
    # Preprocesar
    img_procesada = preprocesar_imagen(ruta_imagen)

    # Predecir
    predicciones = modelo.predict(img_procesada, verbose=0)

    # Obtener la clase con mayor probabilidad
    clase_predicha_idx = np.argmax(predicciones[0])
    clase_predicha = CLASES_IMAGENES[clase_predicha_idx]
    confianza = predicciones[0][clase_predicha_idx]

    # Obtener top 3 predicciones
    top_3_indices = np.argsort(predicciones[0])[-3:][::-1]
    top_3_predicciones = [
        (CLASES_IMAGENES[idx], predicciones[0][idx])
        for idx in top_3_indices
    ]

    return {
        'clase_predicha': clase_predicha,
        'confianza': confianza,
        'top_3': top_3_predicciones,
        'predicciones_raw': predicciones[0]
    }

def predecir_batch(modelo, rutas_imagenes):
    """Hace predicciones sobre múltiples imágenes."""
    resultados = []

    for ruta in rutas_imagenes:
        try:
            resultado = predecir_imagen(modelo, ruta)
            resultado['ruta_imagen'] = ruta
            resultados.append(resultado)
        except Exception as e:
            print(f"Error procesando {ruta}: {e}")
            resultados.append({
                'ruta_imagen': ruta,
                'error': str(e)
            })

    return resultados

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar modelo
    modelo = cargar_modelo("modelo_imagenes.keras")

    # Predecir una imagen
    ruta_imagen = "ruta/a/tu/imagen.jpg"
    if os.path.exists(ruta_imagen):
        resultado = predecir_imagen(modelo, ruta_imagen)

        print(f"Imagen: {ruta_imagen}")
        print(f"Clase predicha: {resultado['clase_predicha']}")
        print(".2%")
        print("\nTop 3 predicciones:")
        for clase, conf in resultado['top_3']:
            print(".2%")
    else:
        print(f"No se encontró la imagen: {ruta_imagen}")
        print("Ejemplo de uso:")
        print("python inferencia_imagenes.py")
        print("Asegúrate de que 'modelo_imagenes.keras' existe en el directorio actual")