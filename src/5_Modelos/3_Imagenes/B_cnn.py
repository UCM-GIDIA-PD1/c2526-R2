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
from sklearn.preprocessing import normalize
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageOps
from tqdm import tqdm
import random
import warnings
import io
import os
import json
import matplotlib.pyplot as plt

def pasar_a_TfRecord(tensor_imagen, etiqueta):
    """Convierte un tensor numpy y un entero en un registro binario de TFRecord."""
    tensor_serializado = tf.io.serialize_tensor(tensor_imagen).numpy()
    
    caracteristicas = {
        'imagen': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_serializado])),
        'etiqueta': tf.train.Feature(int64_list=tf.train.Int64List(value=[etiqueta]))
    }
    
    ejemplo = tf.train.Example(features=tf.train.Features(feature=caracteristicas))
    return ejemplo.SerializeToString()

def descargar_y_preparar_tfrecords(cliente, fases=['train', 'test'], clases=['Cocina', 'Dormitorio', 'Salón', 'Banyo']):
    """
    Descarga desde MinIO, aplica SizeTransformer, balancea las clases dinámicamente,
    aplica Data Augmentation a las clases minoritarias cuando se repiten, 
    y guarda en lotes TFRecord listos para inyectar a la GPU.
    """
    transformador = SizeTransformer()
    base_dir_local = "Dataset_Imagenes"
    os.makedirs(base_dir_local, exist_ok=True)
    
    IMAGENES_POR_TFRECORD = 1024 

    for fase in fases:
        print(f"\n Preprocesando y empaquetando fase: {fase.upper()}")
        dir_fase = os.path.join(base_dir_local, fase)
        os.makedirs(dir_fase, exist_ok=True)

        archivos_minio = {}
        punteros_archivo = {}
        ciclos_completados = {} 
        df_cache = {clase: None for clase in clases}

        for clase in clases:
            path = f"dataset_ml/dataset_vision/{fase}/{clase}"
            archivos_minio[clase] = buscar_todos_los_archivos(cliente, path)
            random.shuffle(archivos_minio[clase]) 
            punteros_archivo[clase] = 0
            ciclos_completados[clase] = 0

        max_archivos_clase_mayoritaria = max([len(archivos) for archivos in archivos_minio.values()])
        

        shard_idx = 0
        lote_balanceado = [] 
        fin_extraccion = False

        barra_progreso = tqdm(desc=f"Generando TFRecords ({fase})")

        while not fin_extraccion:
            for indice_clase, clase in enumerate(clases):
                
                if df_cache[clase] is None or df_cache[clase].empty:
                    
                    if punteros_archivo[clase] >= len(archivos_minio[clase]):
                        punteros_archivo[clase] = 0 
                        ciclos_completados[clase] += 1 
                        random.shuffle(archivos_minio[clase])
                    
                    if ciclos_completados[clase] > 0 and len(archivos_minio[clase]) == max_archivos_clase_mayoritaria:
                        fin_extraccion = True
                        break
                    
                    archivo_actual = archivos_minio[clase][punteros_archivo[clase]]
                    punteros_archivo[clase] += 1
                    
                    path = f"dataset_ml/dataset_vision/{fase}/{clase}"
                    df_temporal = bajar_minio(cliente, path, archivo_actual)
                    
                    df_cache[clase] = df_temporal.sample(frac=1).reset_index(drop=True)

                if fin_extraccion:
                    break

                fila = df_cache[clase].iloc[0]
                df_cache[clase] = df_cache[clase].iloc[1:]

                img = Image.open(io.BytesIO(fila['imagen_bytes']))
                tensor_base = transformador(img) 

                if fase == "train" and ciclos_completados[clase] > 0:
                    tensor_tf = tf.convert_to_tensor(tensor_base, dtype=tf.float32)
                    tensor_aug = aplicar_augmentation(tensor_tf)
                    tensor_base = tf.cast(tensor_aug, tf.uint8).numpy()
                lote_balanceado.append((tensor_base, indice_clase))
            
            if fin_extraccion:
                break
                
            if len(lote_balanceado) >= IMAGENES_POR_TFRECORD:
                random.shuffle(lote_balanceado)
                
                ruta_tfrecord = os.path.join(dir_fase, f"lote_{shard_idx:04d}.tfrecord")
                with tf.io.TFRecordWriter(ruta_tfrecord) as writer:
                    for tensor, etiqueta in lote_balanceado:
                        writer.write(pasar_a_TfRecord(tensor, etiqueta))
                
                barra_progreso.update(1)
                shard_idx += 1
                lote_balanceado = []
            

        if len(lote_balanceado) > 0:
            random.shuffle(lote_balanceado)
            ruta_tfrecord = os.path.join(dir_fase, f"lote_{shard_idx:04d}.tfrecord")
            with tf.io.TFRecordWriter(ruta_tfrecord) as writer:
                for tensor, etiqueta in lote_balanceado:
                    writer.write(pasar_a_TfRecord(tensor, etiqueta))
            barra_progreso.update(1)

        mapa_clases = {str(indice): clase for indice, clase in enumerate(clases)}
    
        ruta_metadatos = os.path.join(base_dir_local, "diccionario_clases.json")
        with open(ruta_metadatos, 'w', encoding='utf-8') as f:
            json.dump(mapa_clases, f, indent=4, ensure_ascii=False)
                
        print(f" Diccionario de clases guardado en: {ruta_metadatos}")

        barra_progreso.close()
        print(f" Fase {fase} completada. Guardados {shard_idx + (1 if len(lote_balanceado) > 0 else 0)} archivos TFRecord.")

def decodificar_TfRecord(ejemplo_serializado):
    """
    Traduce el archivo binario (ceros y unos) de vuelta a un Tensor y un entero.
    """
    diccionario_caracteristicas = {
        'imagen': tf.io.FixedLenFeature([], tf.string),
        'etiqueta': tf.io.FixedLenFeature([], tf.int64)
    }
    
    datos_extraidos = tf.io.parse_single_example(ejemplo_serializado, diccionario_caracteristicas)
    
    imagen_tensor = tf.io.parse_tensor(datos_extraidos['imagen'], out_type=tf.uint8)
    imagen_tensor.set_shape([160, 240, 3]) 
    
    imagen_tensor = tf.cast(imagen_tensor, tf.float32) / 255.0

    etiqueta = tf.cast(datos_extraidos['etiqueta'], tf.int32)
    
    return imagen_tensor, etiqueta

def crear_dataloader_tfrecord(fase, batch_size=32):
    """
    Carga los TFRecords, los parsea y los pasa al modelo.
    """
    carpeta_fase = os.path.join("Dataset_Imagenes", fase)
    patron_archivos = os.path.join(carpeta_fase, "*.tfrecord")
    
    lista_archivos = tf.data.Dataset.list_files(patron_archivos)
    
    dataset = tf.data.TFRecordDataset(lista_archivos, num_parallel_reads=tf.data.AUTOTUNE)
    
    dataset = dataset.map(decodificar_TfRecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    if fase == "train":
        dataset = dataset.shuffle(buffer_size=2000)
        dataset = dataset.repeat() 
        
    dataset = dataset.batch(batch_size)
    
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

class SizeTransformer:
    def __init__(self, target_width=240,target_height = 160, color=(0, 0, 0)):
        self.target_size = (target_width, target_height)
        self.color = color

    def __call__(self, img):

        imagen_rgb = img.convert("RGB")
        imagen_final = ImageOps.pad(imagen_rgb, self.target_size, color=self.color)
        vector = np.array(imagen_final, dtype=np.uint8)        
        return vector
    
def embeddings_cnn_propia(cliente, nombre_modelo_keras,wandb_run_path = "pd1-c2526-team2/CNN_imagenes/jm8fm8mb", batch_size=32):
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
    matriz_embeddings = np.stack(df_final['embedding'].values)
    
    matriz_normalizada = normalize(matriz_embeddings, norm='l2', axis=1)
    
    matriz_normalizada = matriz_normalizada.astype(np.float16)
    
    df_final['embedding'] = list(matriz_normalizada)

    nombre_salida = "embeddings_cnn_propia.parquet"
    
    subir_minio(df_final, cliente, "dataset_ml", nombre_salida)

def configurar_hardware():
    """
    Detecta la disponibilidad de GPU y configura el crecimiento de memoria
    para evitar que TensorFlow crashee por reservar toda la VRAM de golpe.
    """
    print("\n" + "="*40)
    print("  INICIANDO DIAGNÓSTICO DE HARDWARE")
    print("="*40)
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f" Éxito, Se han detectado {len(gpus)} GPU(s) compatibles con CUDA.")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            device_type = "GPU"
        else:
            print("   No se detectó ninguna GPU. TensorFlow usará la CPU.")
            device_type = "CPU"
            
    except Exception as e:
        print(f" Error configurando la GPU: {e}")
        print(" Forzando ejecución en CPU.")
        tf.config.set_visible_devices([], 'GPU') 
        device_type = "CPU"
        
    return device_type

def descargar_todo_minio_a_local(cliente, fases=['train', 'test'], clases=['Cocina', 'Dormitorio', 'Salón', 'Banyo']):
    """
    Replica la estructura de Parquets de MinIO al disco duro local.
    """
    base_dir_local = "datos_locales"
    os.makedirs(base_dir_local, exist_ok=True)
    
    print("\n Iniciando descarga de Imagenes desde MinIO a disco local...")
    
    for fase in fases:
        for clase in clases:
            path_minio = f"dataset_ml/dataset_vision/{fase}/{clase}"
            dir_local_clase = os.path.join(base_dir_local, fase, clase)
            os.makedirs(dir_local_clase, exist_ok=True)
            
            archivos = buscar_todos_los_archivos(cliente, path_minio)
            for archivo in tqdm(archivos, desc=f"Descargando {fase}/{clase}"):
                ruta_guardado = os.path.join(dir_local_clase, archivo)
                
                # Solo descarga si el archivo no existe ya (evita descargas repetidas)
                if not os.path.exists(ruta_guardado):
                    df_chunk = bajar_minio(cliente, path_minio, archivo)
                    df_chunk.to_parquet(ruta_guardado)
                    
    print(" ¡Descarga completada! Los datos están listos en la carpeta 'datos_locales'.")

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


def probar_cnn(configuracion, cliente_minio=None):
    """
    Construye, entrena y evalúa la CNN basándose en la configuración del usuario.
    Maneja automáticamente el tipo de DataLoader necesario.
    """
    clases = ["Salón", "Dormitorio", "Cocina", "Banyo"]
    mi_transformer = SizeTransformer()
    warnings.filterwarnings('ignore')
    BATCH_SIZE = 32

    print("\n" + "="*50)
    print(" INICIANDO PREPARACIÓN DEL ENTRENAMIENTO")
    print("="*50)

    if configuracion['origen_datos'] == 'minio':
            
        dataset_entrenamiento = tf.data.Dataset.from_generator(
            lambda: cargador_datos(cliente_minio, "train", clases, mi_transformer),
            output_signature=(
                tf.TensorSpec(shape=(160, 240, 3), dtype=tf.float32), 
                tf.TensorSpec(shape=(), dtype=tf.int32)             
            )
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        dataset_test = tf.data.Dataset.from_generator(
            lambda: cargador_datos(cliente_minio, "test", clases, mi_transformer),
            output_signature=(
                tf.TensorSpec(shape=(160, 240, 3), dtype=tf.float32), 
                tf.TensorSpec(shape=(), dtype=tf.int32)             
            )
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        pasos_train = 500 
        pasos_test = 100
        
    else: 
        print(" Configurando DataLoaders desde archivos locales...")
        dataset_entrenamiento = crear_dataloader_tfrecord("train", batch_size=BATCH_SIZE)
        dataset_test = crear_dataloader_tfrecord("test", batch_size=BATCH_SIZE)
        
        pasos_train = 1000  
        pasos_test = 200

    print(f"  Construyendo modelo: {configuracion['tipo'].upper()} | Capas: {configuracion['capas']} | Filtros: {configuracion['filtros']}")
    if configuracion["tipo"] == "mejorada":
        modelo = construir_cnn_mejorada(
            cantidad_Layers_convolucion=configuracion["capas"],
            filtros_iniciales=configuracion["filtros"],
            tasa_dropout=configuracion["dropout"]
        )
    else:
        modelo = construir_cnn_basica(
            cantidad_Layers_convolucion=configuracion["capas"],
            filtros_iniciales=configuracion["filtros"],
            tasa_dropout=configuracion["dropout"]
        )

    modelo.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    run = wandb.init(
        entity="pd1-c2526-team2",
        project="CNN_imagenes",
        config=configuracion,
        name=f"CNN_{configuracion['tipo']}_C{configuracion['capas']}_F{configuracion['filtros']}_D{configuracion['dropout']}"
    )

    nombre_archivo = f"mejor_CNN_{configuracion['tipo']}_C{configuracion['capas']}.keras"
    
    guardado_local = ModelCheckpoint(
        filepath=nombre_archivo,
        monitor="val_accuracy", 
        save_best_only=True,   
        mode="max",            
        verbose=1              
    )

    try:
        print("\n Comenzando entrenamiento...")
        modelo.fit(
            dataset_entrenamiento, 
            epochs=150,
            steps_per_epoch=pasos_train,
            validation_data=dataset_test, 
            validation_steps=pasos_test,
            callbacks=[WandbMetricsLogger(), guardado_local]
        )

        print("\n Evaluando el mejor modelo guardado...")
        modelo.load_weights(nombre_archivo) 

        y_true, y_pred = [], []
        
        for lote_imagenes, lote_etiquetas in dataset_test:
            predicciones = modelo.predict(lote_imagenes, verbose=0)
            clases_predichas = np.argmax(predicciones, axis=1)
            y_pred.extend(clases_predichas)
            y_true.extend(lote_etiquetas.numpy())

        test_acc = accuracy_score(y_true, y_pred)
        test_rec = recall_score(y_true, y_pred, average='weighted')
        test_f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f" Resultados Finales -> Accuracy: {test_acc:.4f} | F1-Score: {test_f1:.4f}")

        wandb.log({
            "final_accuracy": test_acc,
            "final_recall": test_rec,
            "final_f1_score": test_f1
        })

    except KeyboardInterrupt:
        print("\n Entrenamiento interrumpido por el usuario.")
    except Exception as e:
        print(f"\n Error durante el entrenamiento: {e}")

    finally:
       wandb.save(nombre_archivo) 
       wandb.finish()
       print("\n Proceso de WandB finalizado y modelo subido.")

def menu_interactivo():
    """
    Menú de consola para configurar el origen de datos y la arquitectura de la CNN.
    """
    configuracion = {}
    
    print("\n" + "="*50)
    print(" SISTEMA DE ENTRENAMIENTO CNN")
    print("="*50)

    print("\n[1] ORIGEN DE DATOS")
    print("  1. Descargar datos a LOCAL (TFRecords) y entrenar")
    print("  2. Entrenar con TFRecords LOCALES ya existentes")
    print("  3. Entrenar leyendo en streaming desde MINIO")
    
    while True:
        op_datos = input("Elige una opción (1-3): ").strip()
        if op_datos == '1':
            configuracion['origen_datos'] = 'descargar_y_local'
            break
        elif op_datos == '2':
            configuracion['origen_datos'] = 'local'
            break
        elif op_datos == '3':
            configuracion['origen_datos'] = 'minio'
            break
        print(" Opción inválida.")

    print("\n[2] ARQUITECTURA DE LA CNN")
    print("Recomendaciones de estructuras:")
    print("  1. Básica    (3 capas, 32 filtros, 0.2 dropout)")
    print("  2. Mejorada  (3 capas, 32 filtros, 0.3 dropout)")
    print("  3. Mejorada+ (4 capas, 64 filtros, 0.5 dropout)")
    print("  4. Personalizada (Configurar manualmente)")
    
    while True:
        op_red = input("Elige una arquitectura (1-4): ").strip()
        if op_red == '1':
            configuracion.update({"tipo": "basica", "capas": 3, "filtros": 32, "dropout": 0.2})
            break
        elif op_red == '2':
            configuracion.update({"tipo": "mejorada", "capas": 3, "filtros": 32, "dropout": 0.3})
            break
        elif op_red == '3':
            configuracion.update({"tipo": "mejorada", "capas": 4, "filtros": 64, "dropout": 0.5})
            break
        elif op_red == '4':
            tipo = input("  ¿Tipo de bloques? (basica/mejorada): ").strip().lower()
            capas = int(input("  ¿Cantidad de capas convolucionales? (ej. 3): "))
            filtros = int(input("  ¿Filtros iniciales? (ej. 32): "))
            dropout = float(input("  ¿Tasa de dropout? (ej. 0.3): "))
            configuracion.update({"tipo": tipo, "capas": capas, "filtros": filtros, "dropout": dropout})
            break
        print(" Opción inválida.")
        
    return configuracion

if __name__ == "__main__":
    embeddings_cnn_propia(crear_cliente_minio(),"mejor_CNN_mejorada_C5.keras")