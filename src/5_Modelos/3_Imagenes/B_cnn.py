import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.utils import img_to_array
from utils.funciones_minio import bajar_minio,crear_cliente_minio,buscar_todos_los_archivos,subir_minio
import wandb
from wandb.integration.keras import WandbMetricsLogger
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from wandb.integration.keras import WandbModelCheckpoint
from sklearn.metrics import f1_score, accuracy_score, recall_score
from PIL import Image, ImageOps
from tqdm import tqdm
import random
import warnings
import io
import matplotlib.pyplot as plt

class SizeTransformer:
    def __init__(self, target_width=240,target_height = 160, color=(0, 0, 0)):
        self.target_size = (target_width, target_height)
        self.color = color

    def __call__(self, img):

        imagen_rgb = img.convert("RGB")
        imagen_final = ImageOps.pad(imagen_rgb, self.target_size, color=self.color)
        vector = np.array(imagen_final) / 255
        return vector
    
def embeddings_cnn_propia(cliente, nombre_modelo_keras,wandb_run_path = "pd1-c2526-team2/CNN_imagenes/4bw0h77i", batch_size=32):
    """
    Descarga un modelo entrenado de WandB, recorta hasta la 'capa_embedings',
    vectoriza todas las imágenes y sube el resultado a MinIO.
    """
    api = wandb.Api()
    run = api.run(wandb_run_path)
    run.file(nombre_modelo_keras).download(replace=True)
    
    modelo_completo = load_model(nombre_modelo_keras)
    
    modelo_extractor = Model(
        inputs=modelo_completo.inputs,
        outputs=modelo_completo.get_layer("capa_embedings").output
    )
    
    clases = ['Cocina', 'Dormitorio', 'Salón', 'Banyo']
    transformador = SizeTransformer() 
    resultados_finales = []
    
    def procesar_lote_keras(vectores, ids, etiquetas):
        if len(vectores) == 0: return
        
        batch_tensor = np.stack(vectores)
        
        embeddings = modelo_extractor.predict(batch_tensor, verbose=0)
        embeddings_np = embeddings.astype(np.float16)
        
        for i in range(len(ids)):
            resultados_finales.append({
                'id': ids[i],
                'clase': etiquetas[i],
                'embedding': embeddings_np[i]
            })

    
    for clase in clases:
        path = f"cleaned/dataset_vision/{clase}"
        objetos = buscar_todos_los_archivos(cliente, path)
        
        for obj in tqdm(objetos, desc=f"Procesando imágenes de {clase}"):
            df_chunk = bajar_minio(cliente, path, obj)
            
            lote_vectores = []
            lote_ids = []
            lote_clases = []
            
            for _, fila in df_chunk.iterrows():
                img = Image.open(io.BytesIO(fila['imagen_bytes']))
                vector_listo = transformador(img)
                    
                lote_vectores.append(vector_listo)
                lote_ids.append(fila['id'])
                lote_clases.append(clase)
                    
                if len(lote_vectores) == batch_size:
                    procesar_lote_keras(lote_vectores, lote_ids, lote_clases)
                    lote_vectores, lote_ids, lote_clases = [], [], []
            
            procesar_lote_keras(lote_vectores, lote_ids, lote_clases)

    df_final = pd.DataFrame(resultados_finales)
    nombre_salida = "embeddings_cnn_propia.parquet"
    
    subir_minio(df_final, cliente, "dataset_ml", nombre_salida)

def aplicar_augmentation(tensor_img):
    img = tf.image.random_flip_left_right(tensor_img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    return img

def cargador_datos(cliente, fase, clases, transformador):
    parquets = {clase: [] for clase in clases}
    archivos_base = {}
    
    for clase in clases:
        path = f"dataset_ml/dataset_vision/{fase}/{clase}"
        archivos_base[clase] = buscar_todos_los_archivos(cliente, path)
        parquets[clase] = archivos_base[clase].copy()

    while True:
        dfs_temporales = []
        
        for indice_num, clase in enumerate(clases):
            if len(parquets[clase]) == 0:
                parquets[clase] = archivos_base[clase].copy()
                random.shuffle(parquets[clase])
            
            archivo = parquets[clase].pop()
            
            path = f"dataset_ml/dataset_vision/{fase}/{clase}"
            df_chunk = bajar_minio(cliente, path, archivo)
            df_chunk['etiqueta_numerica'] = indice_num
            df_chunk['nombre_clase'] = clase
            
            dfs_temporales.append(df_chunk)
            
        df_mezclado = pd.concat(dfs_temporales).sample(frac=1).reset_index(drop=True)
        
        for _, fila in df_mezclado.iterrows():
            img = Image.open(io.BytesIO(fila['imagen_bytes']))
            tensor_listo = transformador(img)            
            
            if fase == "train":
                tensor_listo = aplicar_augmentation(tensor_listo)
            
            yield tensor_listo, fila['etiqueta_numerica']

def construir_cnn_mejorada(cantidad_Layers_convolucion:int,cantidad_embedings=256,filtros_iniciales = 32,tasa_dropout = 0.3,num_clases=4):
    modelo = models.Sequential()

    modelo.add(layers.Conv2D(filtros_iniciales, (3, 3), activation='relu', input_shape=(160, 240, 3)))
    modelo.add(layers.MaxPooling2D((2, 2)))

    filtros_actuales = filtros_iniciales
    for i in range(1, cantidad_Layers_convolucion):
        filtros_actuales *= 2 
        modelo.add(layers.Conv2D(filtros_actuales, (3, 3), padding='same'))
        modelo.add(layers.BatchNormalization())
        modelo.add(layers.Activation('relu'))

        modelo.add(layers.Conv2D(filtros_actuales, (3, 3), padding='same'))
        modelo.add(layers.BatchNormalization())
        modelo.add(layers.Activation('relu'))

        modelo.add(layers.MaxPooling2D((2, 2)))
        
    modelo.add(layers.Flatten())
    
    modelo.add(layers.Dense(cantidad_embedings, activation='relu',name = "capa_embedings"))
    modelo.add(layers.Dropout(tasa_dropout))
        
    modelo.add(layers.Dense(num_clases, activation='softmax',name = "capa_clasificacion"))
    
    return modelo

def construir_cnn_basica(cantidad_Layers_convolucion:int,cantidad_embedings=256,filtros_iniciales = 32,tasa_dropout = 0.3,num_clases=4):
    modelo = models.Sequential()

    modelo.add(layers.Conv2D(filtros_iniciales, (3, 3), activation='relu', input_shape=(160, 240, 3)))
    modelo.add(layers.MaxPooling2D((2, 2)))

    filtros_actuales = filtros_iniciales
    for i in range(1, cantidad_Layers_convolucion):
        filtros_actuales *= 2 
        modelo.add(layers.Conv2D(filtros_actuales, (3, 3), activation='relu'))
        modelo.add(layers.MaxPooling2D((2, 2)))
        
    modelo.add(layers.Flatten())
    
    modelo.add(layers.Dense(cantidad_embedings, activation='relu',name = "capa_embedings"))
    modelo.add(layers.Dropout(tasa_dropout))
        
    modelo.add(layers.Dense(num_clases, activation='softmax',name = "capa_clasificacion"))
    
    return modelo


def probar_distintos_cnn():
    clases = ["Salón","Dormitorio","Cocina","Banyo"]
    mi_transformer = SizeTransformer()
    warnings.filterwarnings('ignore')
    cliente = crear_cliente_minio()
    BATCH_SIZE = 32

    dataset_entrenamiento = tf.data.Dataset.from_generator(
        lambda: cargador_datos(cliente,"train", clases, mi_transformer),
        output_signature=(
            tf.TensorSpec(shape=(160, 240, 3), dtype=tf.float32), 
            tf.TensorSpec(shape=(), dtype=tf.int32)             
        )
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    dataset_test = tf.data.Dataset.from_generator(
        lambda: cargador_datos(cliente,"test", clases, mi_transformer),
        output_signature=(
            tf.TensorSpec(shape=(160, 240, 3), dtype=tf.float32), 
            tf.TensorSpec(shape=(), dtype=tf.int32)             
        )
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    pasos_train = 500
    pasos_test = 1875

    configuraciones_a_probar = [
        {"tipo": "basica", "capas": 3, "filtros": 32, "dropout": 0.2},
    ]

    for config in configuraciones_a_probar:        
        run = wandb.init(
            entity="pd1-c2526-team2",
            project="CNN_imagenes",
            config=config,
            name=f"CNN_{config['tipo']}_C{config['capas']}_F{config['filtros']}"
        )

        if config["tipo"] == "mejorada":
            modelo = construir_cnn_mejorada(
                cantidad_Layers_convolucion=config["capas"],
                filtros_iniciales=config["filtros"],
                tasa_dropout=config["dropout"]
            )
        else:
            modelo = construir_cnn_basica(
                cantidad_Layers_convolucion=config["capas"],
                filtros_iniciales=config["filtros"],
                tasa_dropout=config["dropout"]
            )

        modelo.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        nombre_archivo = f"mejor_CNN_{config['tipo']}_C{config['capas']}_F{config['filtros']}.keras"
        
        guardado_local = ModelCheckpoint(
            filepath=nombre_archivo,
            monitor="accuracy",   
            save_best_only=True,   
            mode="max",            
            verbose=1              
        )

        try :
            modelo.fit(
                dataset_entrenamiento, 
                epochs=150,
                steps_per_epoch=pasos_train,
                callbacks=[WandbMetricsLogger(),guardado_local]
            )

            y_true = []
            y_pred = []
            
            for lote_imagenes, lote_etiquetas in dataset_test.take(pasos_test):
                predicciones = modelo.predict(lote_imagenes, verbose=0)
                clases_predichas = np.argmax(predicciones, axis=1)
                
                y_pred.extend(clases_predichas)
                y_true.extend(lote_etiquetas.numpy())

            test_acc = accuracy_score(y_true, y_pred)
            test_rec = recall_score(y_true, y_pred, average='weighted')
            test_f1 = f1_score(y_true, y_pred, average='weighted')

            
            wandb.log({
                "accuracy": test_acc,
                "recall": test_rec,
                "f1_score": test_f1
            })
        except KeyboardInterrupt:
            print("\n Has parado el entrenamiento.")
        except Exception as e:
            print("\n Error durante el entrenamiento ")

        finally:
           wandb.save(nombre_archivo) 
           wandb.finish()



if __name__ == "__main__":
    cliente = crear_cliente_minio()
    nombre_pryecto = "mejor_CNN_basica_C3_F32.keras"
    embeddings_cnn_propia(cliente,nombre_pryecto)