import tensorflow as tf
from tensorflow.keras import layers, Model
from utils.config import CLASES_IMAGENES
import wandb
from wandb.keras import WandbCallback

# Configuración para que Windows no sufra
AUTOTUNE = tf.data.AUTOTUNE

# Parámetros configurables
IMAGE_SIZE = (160, 240)
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
SEED = 42
EPOCHS = 20
FILTERS = [32, 64, 128]
DENSE_UNITS = 128
OPTIMIZER = 'adam'
LOSS = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']
DATASET_PATH = "dataset_local"

def entrenar():
    # Inicializar wandb
    run = wandb.init(project="image_classification", config={
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "validation_split": VALIDATION_SPLIT,
        "seed": SEED,
        "epochs": EPOCHS,
        "filters": FILTERS,
        "dense_units": DENSE_UNITS,
        "optimizer": OPTIMIZER,
        "loss": LOSS,
        "metrics": METRICS,
        "dataset_path": DATASET_PATH
    })

    # 1. Cargar desde disco de forma eficiente
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # 2. EL TRUCO PARA WINDOWS: Cache y Prefetch
    # .cache() mete las imágenes en RAM tras la primera época. 
    # Con 60k imágenes de este tamaño, te sobrará RAM si tienes 16GB.
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 3. Modelo con GlobalAveragePooling (más rápido que Flatten)
    entrada = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = layers.Rescaling(1./255)(entrada) # Normalización integrada en el modelo
    
    for f in FILTERS:
        x = layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x) # <--- MUCHO más rápido
    x = layers.Dense(DENSE_UNITS, activation='relu', name='embedding')(x)
    salida = layers.Dense(len(CLASES_IMAGENES), activation='softmax')(x)

    modelo = Model(inputs=entrada, outputs=salida)
    modelo.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

    # 4. Entrenar
    # En Windows, evita usar workers=N en el fit si usas tf.data, 
    # el prefetch de arriba ya se encarga de la paralelización.
    modelo.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[WandbCallback()])

    # 5. Guardar el modelo
    modelo.save("modelo_imagenes.keras")
    print("Modelo guardado como 'modelo_imagenes.keras'")

    wandb.finish()

if __name__ == "__main__":
    # IMPORTANTE EN WINDOWS: Todo lo de TF debe ir dentro de este if
    entrenar()