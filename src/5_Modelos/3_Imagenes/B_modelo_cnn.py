# -*- coding: utf-8 -*-
"""B_modelo_cnn.py — CNN de habitaciones con embeddings de 128-d"""

import io
import gc
import numpy as np
import tensorflow as tf
import wandb
from PIL import Image
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix

from utils.funciones_minio import crear_cliente_minio, bajar_minio, buscar_todos_los_archivos
from utils.config import CLASES_IMAGENES, CNN_TARGET_SIZE, CNN_BATCH_SIZE

# Máximo de Parquets por clase (limitar para portátil)
MAX_PARQUETS = 5


# ── Descarga de imágenes desde MinIO ─────────────────────────────────

def descargar_imagenes(clases, max_parquets=MAX_PARQUETS):
    """Baja los primeros N Parquets por clase y devuelve [(bytes, clase_id), ...]"""
    cliente = crear_cliente_minio()
    pool = []
    conteo = {}

    for clase_id, clase in enumerate(clases):
        archivos = buscar_todos_los_archivos(cliente, f"cleaned/dataset_vision/{clase}")
        archivos = archivos[:max_parquets]
        total_clase = 0

        for archivo in archivos:
            df = bajar_minio(cliente, f"cleaned/dataset_vision/{clase}", archivo)
            for raw in df['imagen_bytes']:
                pool.append((bytes(raw), clase_id))
            total_clase += len(df)
            del df

        conteo[clase] = total_clase
        print(f"  {clase}: {total_clase:,} imgs ({len(archivos)} parquets)")

    gc.collect()
    print(f"  Total: {len(pool):,} imagenes")
    return pool, conteo


# ── Generador de batches ─────────────────────────────────────────────

class GeneradorHabitaciones(Sequence):
    """Decodifica JPEGs bajo demanda, sin cargar todo en RAM."""

    def __init__(self, pool, indices, num_clases, batch_size=CNN_BATCH_SIZE, shuffle=True):
        self.pool = pool
        self.indices = indices.copy()
        self.num_clases = num_clases
        self.batch_size = batch_size
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        n = len(batch)
        X = np.empty((n, *CNN_TARGET_SIZE, 3), dtype=np.float32)
        y = np.zeros((n, self.num_clases), dtype=np.float32)

        for i, pos in enumerate(batch):
            jpeg_bytes, clase_id = self.pool[pos]
            img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            img = img.resize((CNN_TARGET_SIZE[1], CNN_TARGET_SIZE[0]))
            X[i] = np.array(img, dtype=np.float32) / 255.0
            y[i, clase_id] = 1.0
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ── Arquitectura CNN ─────────────────────────────────────────────────

def crear_modelo(num_clases, filtros=[32, 64, 128], dim_embedding=128):
    """CNN con bloques Conv2D+BatchNorm+ReLU+MaxPool -> embedding 128-d -> Softmax"""
    entrada = layers.Input(shape=(*CNN_TARGET_SIZE, 3))
    x = entrada

    # Bloques convolucionales
    for f in filtros:
        x = layers.Conv2D(f, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)

    # Embedding
    x = layers.Flatten()(x)
    x = layers.Dense(dim_embedding, activation='relu', name='embedding')(x)

    # Cabeza de clasificación (temporal, solo para entrenar)
    salida = layers.Dense(num_clases, activation='softmax', name='head')(x)

    return Model(inputs=entrada, outputs=salida, name='cnn_habitaciones')


# ── Pesos de clase ───────────────────────────────────────────────────

def calcular_pesos(conteo, clases):
    """Pesos inversamente proporcionales para compensar desbalanceo."""
    cantidades = np.array([conteo[c] for c in clases])
    etiquetas = np.concatenate([np.full(n, i) for i, n in enumerate(cantidades)])
    pesos = compute_class_weight('balanced', classes=np.arange(len(clases)), y=etiquetas)
    for i, c in enumerate(clases):
        print(f"  {c}: peso={pesos[i]:.3f} ({conteo[c]:,} imgs)")
    return dict(enumerate(pesos))


# ══════════════════════════════════════════════════════════════════════
# Flujo principal
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    tf.keras.utils.set_random_seed(42)

    EPOCHS = 20
    FILTROS = [32, 64, 128]
    DIM_EMBEDDING = 128

    # 1. Descargar imágenes
    print("Descargando imagenes desde MinIO...")
    pool, conteo = descargar_imagenes(CLASES_IMAGENES)

    # 2. Split train/val estratificado
    etiquetas = np.array([item[1] for item in pool])
    indices = np.arange(len(pool))
    idx_train, idx_val = train_test_split(indices, test_size=0.2, random_state=42, stratify=etiquetas)

    gen_train = GeneradorHabitaciones(pool, idx_train, len(CLASES_IMAGENES))
    gen_val = GeneradorHabitaciones(pool, idx_val, len(CLASES_IMAGENES), shuffle=False)

    print(f"  Train: {len(idx_train):,} | Val: {len(idx_val):,}")

    # 3. Modelo
    modelo = crear_modelo(len(CLASES_IMAGENES), FILTROS, DIM_EMBEDDING)
    modelo.compile(optimizer='adam', loss='categorical_crossentropy')
    modelo.summary()

    # 4. Pesos de clase
    pesos_clase = calcular_pesos(conteo, CLASES_IMAGENES)

    # 5. W&B
    run = wandb.init(
        entity="pd1-c2526-team2",
        project="clasificador-imagenes",
        name="cnn-embeddings",
        job_type="train",
        config={
            "filtros": FILTROS, "epochs": EPOCHS,
            "target_size": CNN_TARGET_SIZE, "batch_size": CNN_BATCH_SIZE,
            "embedding_dim": DIM_EMBEDDING, "max_parquets": MAX_PARQUETS,
            "clases": CLASES_IMAGENES, "total_imagenes": sum(conteo.values()),
        },
    )

    # 6. Entrenar
    print("Entrenando...")
    modelo.fit(
        gen_train, validation_data=gen_val,
        epochs=EPOCHS, class_weight=pesos_clase,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            wandb.keras.WandbMetricsLogger(),
        ],
    )

    # 7. Matriz de confusión
    y_pred = np.argmax(modelo.predict(gen_val, verbose=1), axis=1)
    y_true = np.concatenate([np.argmax(gen_val[i][1], axis=1) for i in range(len(gen_val))])

    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de Confusión:")
    print(cm)

    for i, c_true in enumerate(CLASES_IMAGENES):
        for j, c_pred in enumerate(CLASES_IMAGENES):
            if i != j and cm[i, j] > 0:
                print(f"  {cm[i,j]} {c_true} -> {c_pred}")

    # 8. Extraer embeddings (modelo sin cabeza)
    modelo_emb = Model(inputs=modelo.input, outputs=modelo.get_layer('embedding').output)
    gen_todo = GeneradorHabitaciones(pool, np.arange(len(pool)), len(CLASES_IMAGENES), shuffle=False)
    embeddings = modelo_emb.predict(gen_todo, verbose=1)

    np.save('embeddings_habitaciones.npy', embeddings)
    np.save('etiquetas.npy', etiquetas)
    print(f"  Embeddings: {embeddings.shape}")

    # 9. Guardar modelo
    modelo.save('modelo_final_habitaciones.keras')
    print("  Modelo guardado: modelo_final_habitaciones.keras")

    wandb.finish()
    print("Completado.")