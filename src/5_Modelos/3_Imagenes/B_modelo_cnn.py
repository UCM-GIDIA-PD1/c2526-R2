"""
B_modelo_cnn.py

Red Neuronal Convolucional (CNN) diseñada desde cero (sin Transfer Learning)
para clasificar imágenes de habitaciones en 4 clases:
Cocina, Dormitorio, Salón y Baño.

Las imágenes se cargan desde MinIO en formato Parquet.
El entrenamiento se registra en Weights & Biases.
"""

import numpy as np
import io
import wandb
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

from utils.funciones_minio import crear_cliente_minio, bajar_minio, buscar_todos_los_archivos
from utils.config import (
    CLASES_IMAGENES,
    MINIO_DATASET_VISION,
    CNN_TARGET_SIZE,
    CNN_BATCH_SIZE,
)

# ─────────────────────────────────────────────────────────────────────
# Generador de datos desde MinIO
# ─────────────────────────────────────────────────────────────────────

class MinioParquetGenerator(Sequence):
    """
    Lee archivos Parquet de MinIO que contienen imágenes en bytes
    y los devuelve como batches (X, y) listos para Keras.
    """

    def __init__(self, cliente, registros, clases, target_size=CNN_TARGET_SIZE):
        self.cliente = cliente
        self.clases = clases
        self.target_size = target_size
        self.mapa_clases = {clase: i for i, clase in enumerate(clases)}
        self.registros = registros
        np.random.shuffle(self.registros)

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, index):
        registro = self.registros[index]

        df = bajar_minio(self.cliente, registro['path'], registro['archivo'])

        X, y = [], []
        clase_id = self.mapa_clases[registro['clase']]

        for _, fila in df.iterrows():
            img = Image.open(io.BytesIO(fila['imagen_bytes'])).convert("RGB")
            img = img.resize(self.target_size)
            X.append(np.array(img) / 255.0)

            etiqueta = np.zeros(len(self.clases))
            etiqueta[clase_id] = 1
            y.append(etiqueta)

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        np.random.shuffle(self.registros)


def crear_generadores(cliente, path_base, clases, val_split=0.2):
    """
    Recoge todos los parquet de MinIO y los divide en
    un generador de entrenamiento y otro de validación.

    Returns:
        (train_gen, val_gen, total_por_clase): Generadores + conteo por clase
    """
    registros = []
    total_por_clase = {clase: 0 for clase in clases}

    for clase in clases:
        ruta_clase = f"{path_base}/{clase}"
        archivos = buscar_todos_los_archivos(cliente, ruta_clase)
        for nombre_archivo in archivos:
            registros.append({
                'path': ruta_clase,
                'archivo': nombre_archivo,
                'clase': clase
            })
        total_por_clase[clase] = len(archivos)

    np.random.seed(42)
    np.random.shuffle(registros)

    punto_corte = int(len(registros) * (1 - val_split))
    train_registros = registros[:punto_corte]
    val_registros = registros[punto_corte:]

    print(f"  Archivos train: {len(train_registros)} | val: {len(val_registros)}")

    train_gen = MinioParquetGenerator(cliente, train_registros, clases)
    val_gen = MinioParquetGenerator(cliente, val_registros, clases)

    return train_gen, val_gen, total_por_clase


# ─────────────────────────────────────────────────────────────────────
# Data Augmentation
# ─────────────────────────────────────────────────────────────────────

def crear_capa_augmentacion():
    """Capa de aumento de datos que se aplica solo durante el entrenamiento."""
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")


# ─────────────────────────────────────────────────────────────────────
# Arquitectura CNN
# ─────────────────────────────────────────────────────────────────────

def construir_cnn(input_shape=(150, 150, 3), num_clases=4):
    """
    CNN de 4 bloques convolucionales con BatchNorm y Data Augmentation.

    Arquitectura:
        Augmentation -> [Conv2D -> BatchNorm -> ReLU -> MaxPool] x4
        -> Flatten -> Dense(256) -> Dropout -> Dense(128) -> Dropout -> Softmax
    """
    model = models.Sequential([
        # Data Augmentation (solo en entrenamiento)
        crear_capa_augmentacion(),

        # Bloque 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloque 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloque 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloque 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Clasificador
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_clases, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
        ]
    )
    return model


# ─────────────────────────────────────────────────────────────────────
# Cálculo automático de pesos de clase
# ─────────────────────────────────────────────────────────────────────

def calcular_pesos_clase(total_por_clase, clases):
    """
    Calcula los pesos de clase automáticamente usando sklearn.
    Las clases con menos ejemplos reciben más peso.
    """
    nombres = list(total_por_clase.keys())
    cantidades = np.array([total_por_clase[c] for c in nombres])

    # Generar etiquetas artificiales proporcionales al número de archivos
    etiquetas = np.concatenate([np.full(n, i) for i, n in enumerate(cantidades)])

    pesos = compute_class_weight('balanced', classes=np.arange(len(clases)), y=etiquetas)
    pesos_dict = {i: peso for i, peso in enumerate(pesos)}

    print("  Pesos de clase calculados:")
    for i, clase in enumerate(clases):
        print(f"    {clase}: {pesos_dict[i]:.2f}")

    return pesos_dict


# ─────────────────────────────────────────────────────────────────────
# Entrenamiento principal
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Configuración
    cliente = crear_cliente_minio()
    epochs = 30

    # 2. Generadores train/val
    print("Creando generadores de datos...")
    train_gen, val_gen, total_por_clase = crear_generadores(
        cliente, MINIO_DATASET_VISION, CLASES_IMAGENES
    )

    # 3. Pesos de clase automáticos
    pesos_clase = calcular_pesos_clase(total_por_clase, CLASES_IMAGENES)

    # 4. Construir modelo
    modelo = construir_cnn(
        input_shape=(*CNN_TARGET_SIZE, 3),
        num_clases=len(CLASES_IMAGENES)
    )
    modelo.summary()

    # 5. W&B
    run = wandb.init(
        entity="pd1-c2526-team2",
        project="clasificador-imagenes",
        name="cnn-from-scratch",
        job_type="train",
        config={
            "arquitectura": "CNN-4bloques",
            "epochs": epochs,
            "target_size": CNN_TARGET_SIZE,
            "batch_size": CNN_BATCH_SIZE,
            "optimizer": "adam",
            "clases": CLASES_IMAGENES,
            "data_augmentation": True,
        }
    )

    # 6. Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath='mejor_modelo_cnn.keras',
            monitor='val_recall',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        wandb.keras.WandbMetricsLogger(),
    ]

    # 7. Entrenar
    print("Iniciando entrenamiento...")
    historia = modelo.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=pesos_clase,
        callbacks=callbacks,
    )

    # 8. Finalizar
    wandb.finish()
    print("Entrenamiento finalizado.")