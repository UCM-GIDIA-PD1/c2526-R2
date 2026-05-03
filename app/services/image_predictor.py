import os
# Suprimir warnings de TensorFlow por consola antes de importarlo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import io
import numpy as np
from PIL import Image, ImageOps
import wandb
import tensorflow as tf
from tensorflow.keras.models import load_model

class SizeTransformer:
    def __init__(self, target_width=240, target_height=160, color=(0, 0, 0)):
        self.target_size = (target_width, target_height)
        self.color = color

    def __call__(self, img: Image.Image) -> np.ndarray:
        imagen_rgb = img.convert("RGB")
        imagen_final = ImageOps.pad(imagen_rgb, self.target_size, color=self.color)
        vector = np.array(imagen_final, dtype=np.uint8)        
        return vector

class ImagePredictor:
    def __init__(self):
        self.model = None
        self.transformer = SizeTransformer()
        self.clases = {
            0: "Cocina",
            1: "Dormitorio",
            2: "Salón",
            3: "Banyo"
        }
        self.wandb_run_path = "pd1-c2526-team2/CNN_imagenes/jm8fm8mb"
        self.model_name = "mejor_CNN_mejorada_C5.keras"
        self.model_cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "model_cache")
        os.makedirs(self.model_cache_dir, exist_ok=True)
        self.local_model_path = os.path.join(self.model_cache_dir, self.model_name)

    def load_model_if_needed(self):
        if self.model is not None:
            return
        
        if not os.path.exists(self.local_model_path):
            print(f"Descargando modelo {self.model_name} desde W&B...")
            api = wandb.Api()
            run = api.run(self.wandb_run_path)
            run.file(self.model_name).download(root=self.model_cache_dir, replace=True)
            print("Descarga completada.")
        
        print("Cargando modelo en memoria...")
        self.model = load_model(self.local_model_path)
        print("Modelo cargado correctamente.")

    def predict_batch(self, image_bytes_list: list[bytes]):
        """
        Recibe una lista de bytes (imágenes), las preprocesa, 
        pasa por el modelo y devuelve la clase mayoritaria y probabilidades.
        """
        self.load_model_if_needed()

        vectores = []
        for img_bytes in image_bytes_list:
            img = Image.open(io.BytesIO(img_bytes))
            vector = self.transformer(img)
            vectores.append(vector)
        
        # Apilar y normalizar como en el TFRecord original (cast a float32 y div 255.0)
        batch_tensor = np.stack(vectores).astype(np.float32) / 255.0

        # Inferencia
        predicciones = self.model.predict(batch_tensor, verbose=0)
        
        resultados = []
        for prob_array in predicciones:
            clase_idx = int(np.argmax(prob_array))
            clase_nombre = self.clases.get(clase_idx, "Desconocido")
            
            # Formatear probabilidades
            probabilidades = {
                self.clases[i]: float(prob_array[i])
                for i in range(len(prob_array))
            }
            
            resultados.append({
                "clase": clase_nombre,
                "probabilidades": probabilidades
            })
            
        return resultados

# Instancia singleton para ser usada por el router
predictor = ImagePredictor()
