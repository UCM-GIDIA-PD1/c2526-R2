"""
B_modelo_cnn.py

Red Neuronal Convolucional (CNN) diseñada desde cero (sin Transfer Learning)
para clasificar imágenes de habitaciones en 4 clases:
Cocina, Dormitorio, Salón y Baño.

Las imágenes se cargan desde MinIO en formato Parquet.
El entrenamiento se registra en Weights & Biases.

Optimizaciones respecto a la versión anterior:
─────────────────────────────────────────────────
 • Caché completo en memoria de los bytes JPEG (~5 GB para 172K imgs)
 • Decodificación lazy: solo se decodifican las imágenes del batch actual
 • Batches de tamaño constante (CNN_BATCH_SIZE) independiente del Parquet
 • Shuffle a nivel de imagen individual (no por fichero Parquet)
 • Split train/val estratificado por clase
 • Class weights calculados por nº real de imágenes (no por nº de ficheros)
 • Workers con threading para solapar decode CPU ↔ GPU training
"""

import gc
import numpy as np
import io
import wandb
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
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
# Generador de datos optimizado
# ─────────────────────────────────────────────────────────────────────

class MinioParquetGenerator(Sequence):
    """
    Generador Keras que entrega batches de tamaño constante desde
    un pool de imágenes JPEG cacheadas en memoria.

    Las imágenes se almacenan como bytes JPEG comprimidos y se
    decodifican a numpy solo en el momento del batch (lazy decode).

    Ambos generadores (train/val) comparten la misma referencia al
    pool de imágenes, usando índices distintos, sin duplicar memoria.
    """

    def __init__(self, imagenes_pool, indices, clases,
                 batch_size=CNN_BATCH_SIZE, target_size=CNN_TARGET_SIZE):
        """
        Args:
            imagenes_pool: Lista compartida de (jpeg_bytes, clase_id).
                           NO se copia, se usa por referencia.
            indices: Array de índices asignados a este generador.
            clases: Lista de nombres de clase.
            batch_size: Tamaño del batch (constante).
            target_size: (alto, ancho) destino de cada imagen.
        """
        self.imagenes = imagenes_pool
        self.indices = indices.copy()
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_clases = len(clases)
        np.random.shuffle(self.indices)

    def __len__(self):
        """Nº de batches por epoch (constante, no depende del Parquet)."""
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        """Decodifica solo las imágenes del batch `idx`."""
        inicio = idx * self.batch_size
        fin = min(inicio + self.batch_size, len(self.indices))
        batch_indices = self.indices[inicio:fin]
        tamano = fin - inicio

        # Pre-alocar arrays para evitar append + conversión
        X = np.empty((tamano, *self.target_size, 3), dtype=np.float16)
        y = np.zeros((tamano, self.num_clases), dtype=np.float32)

        for i, img_idx in enumerate(batch_indices):
            jpeg_bytes, clase_id = self.imagenes[img_idx]

            # Decode lazy: JPEG → PIL → resize → float32 normalizado
            img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            img = img.resize(self.target_size)
            X[i] = np.array(img, dtype=np.float16) / 255.0
            y[i, clase_id] = 1.0

        return X, y

    def on_epoch_end(self):
        """Re-shuffle del índice a nivel de imagen individual."""
        np.random.shuffle(self.indices)


# ─────────────────────────────────────────────────────────────────────
# Precarga de datos desde MinIO
# ─────────────────────────────────────────────────────────────────────

def precargar_imagenes_minio(cliente, path_base, clases):
    """
    Descarga todos los Parquets de MinIO, extrae exclusivamente
    los bytes JPEG y construye un pool en memoria.

    Solo descarga la columna 'imagen_bytes' para minimizar
    uso de red y memoria durante el parsing de Parquet.

    Returns:
        imagenes: Lista de (jpeg_bytes, clase_id) — ~5 GB para 172K imgs JPEG
        total_por_clase: Dict {clase: nº_imágenes_real}
    """
    mapa_clases = {clase: i for i, clase in enumerate(clases)}
    imagenes = []
    total_por_clase = {clase: 0 for clase in clases}

    # Descubrir todos los archivos Parquet por clase
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
        # Solo la columna de imagen → ahorra RAM y tiempo de parsing
        df = bajar_minio_especifico(
            cliente, reg['path'], reg['archivo'], ['imagen_bytes']
        )
        clase_id = mapa_clases[reg['clase']]
        n_imgs = len(df)

        for raw_bytes in df['imagen_bytes']:
            imagenes.append((bytes(raw_bytes), clase_id))

        total_por_clase[reg['clase']] += n_imgs

        # Liberar el DataFrame inmediatamente
        del df

        if (i + 1) % 10 == 0 or i == len(registros) - 1:
            print(f"    [{i+1}/{len(registros)}] — {len(imagenes):,} imágenes cargadas")

    # Forzar limpieza de buffers internos de pyarrow/pandas
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
    Descarga todos los datos, cuenta imágenes reales y crea
    generadores train/val con batches de tamaño constante.

    El split se realiza a nivel de imagen individual con
    estratificación por clase (misma proporción en train y val).

    Returns:
        (train_gen, val_gen, total_por_clase)
    """
    imagenes, total_por_clase = precargar_imagenes_minio(
        cliente, path_base, clases
    )

    # Split estratificado por clase a nivel de imagen
    etiquetas = np.array([img[1] for img in imagenes])
    indices = np.arange(len(imagenes))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=42,
        stratify=etiquetas,
    )

    train_gen = MinioParquetGenerator(
        imagenes, train_idx, clases, batch_size
    )
    val_gen = MinioParquetGenerator(
        imagenes, val_idx, clases, batch_size
    )

    print(f"\n  Imágenes  train: {len(train_idx):,} | val: {len(val_idx):,}")
    print(f"  Batches   train: {len(train_gen):,} | val: {len(val_gen):,}")
    print(f"  Batch size: {batch_size}")

    return train_gen, val_gen, total_por_clase


# ─────────────────────────────────────────────────────────────────────
# Data Augmentation (ejecutada en GPU dentro del grafo TF)
# ─────────────────────────────────────────────────────────────────────

def crear_capa_augmentacion():
    """Capa de aumento de datos ejecutada en GPU (solo durante training=True)."""
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
# Cálculo automático de pesos de clase (por nº real de imágenes)
# ─────────────────────────────────────────────────────────────────────

def calcular_pesos_clase(total_por_clase, clases):
    """
    Calcula los pesos inversamente proporcionales al
    nº REAL de imágenes por clase (no nº de ficheros).
    """
    cantidades = np.array([total_por_clase[c] for c in clases])

    # Generar etiquetas sintéticas proporcionales a los conteos reales
    etiquetas = np.concatenate(
        [np.full(n, i) for i, n in enumerate(cantidades)]
    )

    pesos = compute_class_weight(
        'balanced', classes=np.arange(len(clases)), y=etiquetas
    )
    pesos_dict = {i: peso for i, peso in enumerate(pesos)}

    print("  Pesos de clase (por nº real de imágenes):")
    for i, clase in enumerate(clases):
        print(f"    {clase}: {pesos_dict[i]:.4f}  ({total_por_clase[clase]:,} imgs)")

    return pesos_dict


# ─────────────────────────────────────────────────────────────────────
# Entrenamiento principal
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Semillas globales para reproducibilidad
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)

    # 1. Configuración
    cliente = crear_cliente_minio()
    epochs = 30

    # 2. Precarga completa en memoria + generadores con batches constantes
    print("Precargando imágenes desde MinIO...")
    train_gen, val_gen, total_por_clase = crear_generadores(
        cliente, MINIO_DATASET_VISION, CLASES_IMAGENES
    )

    # 3. Pesos de clase por nº real de imágenes
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

    # 7. Entrenar con workers para solapar decode CPU ↔ GPU
    #    use_multiprocessing=False → threading (comparte memoria del pool)
    print("Iniciando entrenamiento...")
    historia = modelo.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=pesos_clase,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=False,
        max_queue_size=16,
    )

    # 8. Finalizar
    wandb.finish()
    print("Entrenamiento finalizado.")