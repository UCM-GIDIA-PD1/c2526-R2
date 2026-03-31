"""
B_modelo_cnn.py

CNN que genera embeddings de 128-d para imágenes de habitaciones.
Se entrena como clasificador (Softmax temporal) y después se le quita
la cabeza para quedarnos solo con la capa Dense(128).
"""

import gc
import io

import numpy as np
import wandb
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models, Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix

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


# --- Generador de datos ---

class GeneradorHabitaciones(Sequence):
    """
    Genera batches decodificando JPEGs bajo demanda.
    Mantiene los bytes comprimidos en RAM y solo convierte
    a numpy el batch que toca, así no petamos la memoria.
    """

    def __init__(self, pool_habitaciones, indices, clases,
                 batch_size=CNN_BATCH_SIZE, target_size=CNN_TARGET_SIZE,
                 shuffle=True):
        self.pool = pool_habitaciones
        self.indices = indices.copy()
        self.batch_size = batch_size
        self.target_size = target_size  # (alto, ancho) -> (160, 240)
        self.num_clases = len(clases)
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        inicio = idx * self.batch_size
        fin = min(inicio + self.batch_size, len(self.indices))
        batch_idx = self.indices[inicio:fin]
        n = fin - inicio

        X = np.empty((n, *self.target_size, 3), dtype=np.float32)
        y = np.zeros((n, self.num_clases), dtype=np.float32)

        for i, pos in enumerate(batch_idx):
            jpeg_bytes, clase_id = self.pool[pos]
            img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            img = img.resize((self.target_size[1], self.target_size[0]))
            X[i] = np.array(img, dtype=np.float32) / 255.0
            y[i, clase_id] = 1.0

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# --- Carga desde MinIO ---

def descargar_pool_habitaciones(cliente, path_base, clases):
    """
    Baja todos los Parquets y devuelve una lista de (jpeg_bytes, clase_id).
    Liberamos cada DataFrame justo después de extraer los bytes.
    """
    mapa_clases = {clase: i for i, clase in enumerate(clases)}
    pool_habitaciones = []
    conteo_clase = {c: 0 for c in clases}

    registros = []
    for clase in clases:
        ruta = f"{path_base}/{clase}"
        for archivo in buscar_todos_los_archivos(cliente, ruta):
            registros.append({'path': ruta, 'archivo': archivo, 'clase': clase})

    print(f"  Descargando {len(registros)} Parquets...")

    for i, reg in enumerate(registros):
        df = bajar_minio_especifico(
            cliente, reg['path'], reg['archivo'], ['imagen_bytes']
        )
        clase_id = mapa_clases[reg['clase']]

        for raw in df['imagen_bytes']:
            pool_habitaciones.append((bytes(raw), clase_id))

        conteo_clase[reg['clase']] += len(df)
        del df

        if (i + 1) % 10 == 0 or i == len(registros) - 1:
            print(f"    [{i+1}/{len(registros)}] {len(pool_habitaciones):,} imgs")

    gc.collect()

    print(f"  Total en cache: {len(pool_habitaciones):,} imagenes")
    for clase, n in conteo_clase.items():
        print(f"    {clase}: {n:,}")

    return pool_habitaciones, conteo_clase


def crear_generadores(cliente, path_base, clases,
                      val_split=0.2, batch_size=CNN_BATCH_SIZE):
    """Split estratificado y creación de generadores train/val."""
    pool_habitaciones, conteo_clase = descargar_pool_habitaciones(
        cliente, path_base, clases
    )

    etiquetas = np.array([item[1] for item in pool_habitaciones])
    indices = np.arange(len(pool_habitaciones))

    idx_train, idx_val = train_test_split(
        indices, test_size=val_split, random_state=42, stratify=etiquetas
    )

    gen_train = GeneradorHabitaciones(pool_habitaciones, idx_train, clases, batch_size)
    gen_val = GeneradorHabitaciones(pool_habitaciones, idx_val, clases, batch_size)

    print(f"\n  Train: {len(idx_train):,} imgs ({len(gen_train)} batches)")
    print(f"  Val:   {len(idx_val):,} imgs ({len(gen_val)} batches)")

    return gen_train, gen_val, pool_habitaciones, conteo_clase


# --- Arquitectura CNN modular ---

def crear_bloque_conv(x, filtros, kernel=(3, 3)):
    """
    Conv2D + BatchNorm + ReLU + MaxPool.
    Usamos BN para evitar que los gradientes se mueran en redes
    más profundas y para estabilizar el entrenamiento en general.
    """
    x = layers.Conv2D(filtros, kernel, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x


def construir_modelo(input_shape, filtros_por_bloque=(32, 64, 128),
                     dim_embedding=128):
    """
    Construye la CNN apilando N bloques convolucionales.

    Parámetros:
        filtros_por_bloque: tupla con los filtros de cada bloque.
            Ej: (32, 64) para una red ligera, (32, 64, 128, 256) si
            quisiéramos más capacidad. Lo dejamos configurable para
            poder comparar variantes sin tocar código.
        dim_embedding: tamaño del vector de salida.

    Devuelve un Model de Keras cuya última capa es Dense(dim_embedding).
    """
    entrada = layers.Input(shape=input_shape)
    x = entrada

    for n_filtros in filtros_por_bloque:
        x = crear_bloque_conv(x, n_filtros)

    # Aplanar y proyectar al espacio de embeddings
    x = layers.Flatten()(x)
    salida = layers.Dense(dim_embedding, activation='relu', name='embedding')(x)

    return Model(inputs=entrada, outputs=salida, name='cnn_habitaciones')


# --- Pesos de clase ---

def calcular_pesos_clase(conteo_clase, clases):
    """Pesos inversamente proporcionales al número de imágenes por clase."""
    cantidades = np.array([conteo_clase[c] for c in clases])
    etiquetas_dummy = np.concatenate(
        [np.full(n, i) for i, n in enumerate(cantidades)]
    )
    pesos = compute_class_weight(
        'balanced', classes=np.arange(len(clases)), y=etiquetas_dummy
    )
    pesos_dict = dict(enumerate(pesos))

    print("  Pesos de clase:")
    for i, clase in enumerate(clases):
        print(f"    {clase}: {pesos_dict[i]:.4f}  ({conteo_clase[clase]:,} imgs)")

    return pesos_dict


# --- Extracción de embeddings batch a batch ---

def extraer_embeddings(modelo_embedding, pool_habitaciones, clases,
                       batch_size=CNN_BATCH_SIZE):
    """
    Pasa todas las imágenes por el modelo usando el generador
    (batch a batch) para no cargar todo en RAM de golpe.
    Devuelve arrays de embeddings y etiquetas.
    """
    gen_completo = GeneradorHabitaciones(
        pool_habitaciones,
        np.arange(len(pool_habitaciones)),
        clases,
        batch_size=batch_size,
        shuffle=False,  # importante: orden consistente con las etiquetas
    )

    # predict ya itera batch a batch internamente
    embeddings = modelo_embedding.predict(gen_completo, verbose=1)
    etiquetas = np.array([item[1] for item in pool_habitaciones])

    return embeddings, etiquetas


# =================================================================
# Flujo principal: Configuración -> Carga -> Entrenamiento -> Extracción
# =================================================================

if __name__ == "__main__":

    # -- Configuración --
    tf.keras.utils.set_random_seed(42)
    cliente = crear_cliente_minio()

    EPOCHS = 20
    FILTROS = [32, 64, 128]  # probar tambien [32, 64] o [64, 128, 256]
    DIM_EMBEDDING = 128

    # -- Carga de datos --
    print("Descargando imagenes desde MinIO...")
    gen_train, gen_val, pool_habitaciones, conteo_clase = crear_generadores(
        cliente, MINIO_DATASET_VISION, CLASES_IMAGENES
    )

    # -- Preprocesado: pesos de clase --
    pesos_clase = calcular_pesos_clase(conteo_clase, CLASES_IMAGENES)

    # -- Construcción del modelo --
    modelo_base = construir_modelo(
        input_shape=(*CNN_TARGET_SIZE, 3),
        filtros_por_bloque=FILTROS,
        dim_embedding=DIM_EMBEDDING,
    )

    # Cabeza Softmax temporal: solo sirve para entrenar como clasificador
    salida_clasificador = layers.Dense(
        len(CLASES_IMAGENES), activation='softmax', name='head_temporal'
    )(modelo_base.output)

    modelo_entrenamiento = Model(
        inputs=modelo_base.input,
        outputs=salida_clasificador,
        name='cnn_clasificador_temporal',
    )
    modelo_entrenamiento.compile(optimizer='adam', loss='categorical_crossentropy')
    modelo_entrenamiento.summary()

    # -- W&B --
    run = wandb.init(
        entity="pd1-c2526-team2",
        project="clasificador-imagenes",
        name="cnn-embeddings",
        job_type="train",
        config={
            "arquitectura": f"CNN-{len(FILTROS)}bloques-BN",
            "filtros": FILTROS,
            "epochs": EPOCHS,
            "target_size": CNN_TARGET_SIZE,
            "batch_size": CNN_BATCH_SIZE,
            "optimizer": "adam",
            "embedding_dim": DIM_EMBEDDING,
            "clases": CLASES_IMAGENES,
            "total_imagenes": sum(conteo_clase.values()),
            "imagenes_por_clase": conteo_clase,
        },
    )

    # -- Entrenamiento --
    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1,
        ),
        wandb.keras.WandbMetricsLogger(),
    ]

    print("Iniciando entrenamiento...")
    modelo_entrenamiento.fit(
        gen_train,
        validation_data=gen_val,
        epochs=EPOCHS,
        class_weight=pesos_clase,
        callbacks=callbacks,
    )

    # -- Confusion Matrix --
    print("\nCalculando matriz de confusión...")
    y_pred = modelo_entrenamiento.predict(gen_val, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    y_true = []
    for i in range(len(gen_val)):
        _, y_batch = gen_val[i]
        y_true.extend(np.argmax(y_batch, axis=1))
    y_true = np.array(y_true)

    cm = confusion_matrix(y_true, y_pred_classes)
    print("Matriz de Confusión:")
    print(cm)

    # Imprimir confusiones específicas
    for i, clase_true in enumerate(CLASES_IMAGENES):
        for j, clase_pred in enumerate(CLASES_IMAGENES):
            if i != j and cm[i, j] > 0:
                print(f"Se han confundido {cm[i, j]} {clase_true} con {clase_pred}")

    # -- Extracción de embeddings --
    # modelo_base ya termina en Dense(128), no hace falta pop()
    print(f"\nModelo de embeddings: salida = {modelo_base.output_shape}")

    embeddings, etiquetas = extraer_embeddings(
        modelo_base, pool_habitaciones, CLASES_IMAGENES
    )

    np.save('embeddings_habitaciones.npy', embeddings)
    np.save('etiquetas.npy', etiquetas)

    print(f"\n  embeddings_habitaciones.npy -> {embeddings.shape}")
    print(f"  etiquetas.npy              -> {etiquetas.shape}")
    print(f"  Clases: {CLASES_IMAGENES}")

    # -- Guardar modelo --
    modelo_entrenamiento.save('modelo_final_habitaciones.keras')

    # -- Fin --
    wandb.finish()
    print("Entrenamiento y extraccion completados.")