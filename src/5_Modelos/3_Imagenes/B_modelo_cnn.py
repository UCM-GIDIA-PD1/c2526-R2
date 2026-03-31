"""
B_modelo_cnn.py

Red Neuronal Convolucional (CNN) que genera embeddings de 128 dimensiones
a partir de imágenes de habitaciones (Cocina, Dormitorio, Salón, Baño).

Estrategia:
───────────
 1. Se entrena una CNN con clasificación normal (Softmax).
 2. Tras el entrenamiento, se extrae la capa Dense(128) como
    vector de representación (embedding) de cada imagen.
 3. Se guardan los embeddings en embeddings_habitaciones.npy
    y las etiquetas en etiquetas.npy.

Las imágenes se cargan desde MinIO en formato Parquet.
El entrenamiento se registra en Weights & Biases.
"""

import gc
import io

import numpy as np
import wandb
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from utils.funciones_minio import (
    crear_cliente_minio,
    bajar_minio_especifico,
    buscar_todos_los_archivos,
)
from utils.config import (
    CLASES_IMAGENES,
    MINIO_DATASET_VISION,
    CNN_TARGET_SIZE,
    CNN_BATCH_SIZE,
)


# ─────────────────────────────────────────────────────────────────────
# Generador de datos
# ─────────────────────────────────────────────────────────────────────

class MinioParquetGenerator(Sequence):
    """
    Generador Keras que entrega batches de imágenes desde memoria.

    Las imágenes se almacenan como bytes JPEG comprimidos y se
    decodifican a numpy solo cuando se necesitan (lazy decode).
    """

    def __init__(self, imagenes_pool, indices, clases,
                 batch_size=CNN_BATCH_SIZE, target_size=CNN_TARGET_SIZE):
        self.imagenes = imagenes_pool
        self.indices = indices.copy()
        self.batch_size = batch_size
        self.target_size = target_size      # (alto, ancho) → (160, 240)
        self.num_clases = len(clases)
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        inicio = idx * self.batch_size
        fin = min(inicio + self.batch_size, len(self.indices))
        batch_indices = self.indices[inicio:fin]
        tamano = fin - inicio

        X = np.empty((tamano, *self.target_size, 3), dtype=np.float32)
        y = np.zeros((tamano, self.num_clases), dtype=np.float32)

        for i, img_idx in enumerate(batch_indices):
            jpeg_bytes, clase_id = self.imagenes[img_idx]

            img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            img = img.resize((self.target_size[1], self.target_size[0]))
            X[i] = np.array(img, dtype=np.float32) / 255.0
            y[i, clase_id] = 1.0

        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# ─────────────────────────────────────────────────────────────────────
# Precarga de datos desde MinIO
# ─────────────────────────────────────────────────────────────────────

def precargar_imagenes_minio(cliente, path_base, clases):
    """
    Descarga todos los Parquets desde MinIO y construye un pool
    en memoria de (jpeg_bytes, clase_id).
    """
    mapa_clases = {clase: i for i, clase in enumerate(clases)}
    imagenes = []
    total_por_clase = {clase: 0 for clase in clases}

    registros = []
    for clase in clases:
        ruta_clase = f"{path_base}/{clase}"
        archivos = buscar_todos_los_archivos(cliente, ruta_clase)
        for nombre in archivos:
            registros.append({
                'path': ruta_clase,
                'archivo': nombre,
                'clase': clase,
            })

    print(f"  Descargando {len(registros)} archivos Parquet...")

    for i, reg in enumerate(registros):
        df = bajar_minio_especifico(
            cliente, reg['path'], reg['archivo'], ['imagen_bytes']
        )
        clase_id = mapa_clases[reg['clase']]

        for raw_bytes in df['imagen_bytes']:
            imagenes.append((bytes(raw_bytes), clase_id))

        total_por_clase[reg['clase']] += len(df)
        del df

        if (i + 1) % 10 == 0 or i == len(registros) - 1:
            print(f"    [{i+1}/{len(registros)}] — {len(imagenes):,} imágenes cargadas")

    gc.collect()

    print(f"  ✅ Total en caché: {len(imagenes):,} imágenes")
    for clase, n in total_por_clase.items():
        print(f"    {clase}: {n:,}")

    return imagenes, total_por_clase


# ─────────────────────────────────────────────────────────────────────
# Creación de generadores con split estratificado
# ─────────────────────────────────────────────────────────────────────

def crear_generadores(cliente, path_base, clases,
                      val_split=0.2, batch_size=CNN_BATCH_SIZE):
    """
    Descarga todos los datos y crea generadores train/val
    con split estratificado por clase.
    """
    imagenes, total_por_clase = precargar_imagenes_minio(
        cliente, path_base, clases
    )

    etiquetas = np.array([img[1] for img in imagenes])
    indices = np.arange(len(imagenes))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=42,
        stratify=etiquetas,
    )

    train_gen = MinioParquetGenerator(imagenes, train_idx, clases, batch_size)
    val_gen = MinioParquetGenerator(imagenes, val_idx, clases, batch_size)

    print(f"\n  Imágenes  train: {len(train_idx):,} | val: {len(val_idx):,}")
    print(f"  Batches   train: {len(train_gen):,} | val: {len(val_gen):,}")
    print(f"  Batch size: {batch_size}")

    return train_gen, val_gen, imagenes, total_por_clase


# ─────────────────────────────────────────────────────────────────────
# Arquitectura CNN
# ─────────────────────────────────────────────────────────────────────

def construir_cnn(input_shape, num_clases=4):
    """
    CNN sencilla de 3 bloques convolucionales.

    Arquitectura:
        [Conv2D → ReLU → MaxPool] × 3
        → Flatten → Dense(128, name='embedding') → Softmax

    La capa Dense(128) actúa como embedding: tras entrenar, se
    puede extraer su salida como vector de representación.
    """
    model = models.Sequential([
        # Bloque 1: 32 filtros
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Bloque 2: 64 filtros
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Bloque 3: 128 filtros
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Capa de embedding (128 dimensiones)
        layers.Flatten(),
        layers.Dense(128, activation='relu', name='embedding'),

        # Clasificador (se usará solo durante el entrenamiento)
        layers.Dense(num_clases, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
    )
    return model


# ─────────────────────────────────────────────────────────────────────
# Cálculo de pesos de clase
# ─────────────────────────────────────────────────────────────────────

def calcular_pesos_clase(total_por_clase, clases):
    """
    Calcula pesos inversamente proporcionales al nº real de imágenes
    por clase para compensar desbalanceo.
    """
    cantidades = np.array([total_por_clase[c] for c in clases])

    etiquetas = np.concatenate(
        [np.full(n, i) for i, n in enumerate(cantidades)]
    )

    pesos = compute_class_weight(
        'balanced', classes=np.arange(len(clases)), y=etiquetas
    )
    pesos_dict = {i: peso for i, peso in enumerate(pesos)}

    print("  Pesos de clase:")
    for i, clase in enumerate(clases):
        print(f"    {clase}: {pesos_dict[i]:.4f}  ({total_por_clase[clase]:,} imgs)")

    return pesos_dict


# ─────────────────────────────────────────────────────────────────────
# Decodificar todas las imágenes a un array numpy
# ─────────────────────────────────────────────────────────────────────

def decodificar_todas(imagenes_pool, target_size=CNN_TARGET_SIZE):
    """
    Convierte el pool completo de JPEG bytes a un array (N, H, W, 3)
    normalizado a [0, 1], junto con sus etiquetas.
    """
    n = len(imagenes_pool)
    X = np.empty((n, *target_size, 3), dtype=np.float32)
    y = np.empty(n, dtype=np.int32)

    for i, (jpeg_bytes, clase_id) in enumerate(imagenes_pool):
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        img = img.resize((target_size[1], target_size[0]))
        X[i] = np.array(img, dtype=np.float32) / 255.0
        y[i] = clase_id

    return X, y


# ─────────────────────────────────────────────────────────────────────
# Entrenamiento principal
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Semilla para reproducibilidad
    tf.keras.utils.set_random_seed(42)

    # 1. Configuración
    cliente = crear_cliente_minio()
    epochs = 20

    # 2. Precarga completa en memoria + generadores
    print("Precargando imágenes desde MinIO...")
    train_gen, val_gen, imagenes_pool, total_por_clase = crear_generadores(
        cliente, MINIO_DATASET_VISION, CLASES_IMAGENES
    )

    # 3. Pesos de clase
    pesos_clase = calcular_pesos_clase(total_por_clase, CLASES_IMAGENES)

    # 4. Construir modelo
    modelo = construir_cnn(
        input_shape=(*CNN_TARGET_SIZE, 3),
        num_clases=len(CLASES_IMAGENES),
    )
    modelo.summary()

    # 5. W&B
    run = wandb.init(
        entity="pd1-c2526-team2",
        project="clasificador-imagenes",
        name="cnn-embeddings",
        job_type="train",
        config={
            "arquitectura": "CNN-3bloques-embeddings-128d",
            "epochs": epochs,
            "target_size": CNN_TARGET_SIZE,
            "batch_size": CNN_BATCH_SIZE,
            "optimizer": "adam",
            "embedding_dim": 128,
            "clases": CLASES_IMAGENES,
            "total_imagenes": sum(total_por_clase.values()),
            "imagenes_por_clase": total_por_clase,
        }
    )

    # 6. Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        wandb.keras.WandbMetricsLogger(),
    ]

    # 7. Entrenar
    print("Iniciando entrenamiento...")
    modelo.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=pesos_clase,
        callbacks=callbacks,

    )

    # ─────────────────────────────────────────────────────────────────
    # 8. Extraer embeddings
    #
    #    Creamos un modelo "extractor" que reutiliza las capas ya
    #    entrenadas, pero cuya salida es la capa Dense(128) en
    #    lugar del Softmax final.
    # ─────────────────────────────────────────────────────────────────
    extractor = models.Model(
        inputs=modelo.input,
        outputs=modelo.get_layer('embedding').output,
    )

    print("\nDecodificando todas las imágenes...")
    X_todas, etiquetas = decodificar_todas(imagenes_pool)

    print("Extrayendo embeddings...")
    embeddings = extractor.predict(X_todas, batch_size=CNN_BATCH_SIZE)

    # 9. Guardar como .npy
    np.save('embeddings_habitaciones.npy', embeddings)
    np.save('etiquetas.npy', etiquetas)

    print(f"\n  ✅ embeddings_habitaciones.npy  → shape {embeddings.shape}")
    print(f"  ✅ etiquetas.npy               → shape {etiquetas.shape}")
    print(f"  Clases: {CLASES_IMAGENES}")

    # 10. Finalizar
    wandb.finish()
    print("Entrenamiento y extracción de embeddings finalizado.")