import os
import joblib
import numpy as np
import tensorflow as tf
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from utils.funciones_minio import crear_cliente_minio, bajar_minio
from utils.config import MINIO_EMBEDDINGS

# =========================================================
# PARÁMETROS CONFIGURABLES
# =========================================================
# Selecciona el archivo de embeddings (ej. "embeddings_imagenes.parquet" o "embeddings_cnn_propia.parquet")
ARCHIVO_EMBEDDINGS = "embeddings_imagenes.parquet"

ARQUITECTURA = (1024, 512)
LEARNING_RATE_INIT = 0.0001
DROPOUT = 0.3
EPOCHS = 40
BATCH_SIZE = 128
PATIENCE = 5
# =========================================================

CLASES_PERMITIDAS = ['Cocina', 'Dormitorio', 'Salón', 'Banyo']

def cargar_y_preparar_datos(archivo_embeddings=ARCHIVO_EMBEDDINGS):
    """
    Descarga los embeddings de MinIO, filtra por clases permitidas,
    aplica PCA para reducción dimensional, codifica las etiquetas y divide en train/test.
    """
    cliente = crear_cliente_minio()
    df = bajar_minio(cliente, MINIO_EMBEDDINGS, archivo_embeddings)
    
    # Filtrar solo por las clases permitidas
    df = df[df['clase'].isin(CLASES_PERMITIDAS)]
    
    # Extraer variables X e y
    X = np.stack(df["embedding"].values)
    y_raw = df["clase"].values
    
    # Codificar la variable objetivo (y) de string a numérico
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)

    # División de datos de forma estratificada train/test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=101, stratify=y_encoded
    )

    # Split estratificado adicional para obtener validación
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    # Aplicar PCA ajustándose solo al Train Set para evitar Data Leakage
    max_components = min(512, min(X_train.shape))
    pca = PCA(n_components=max_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, num_classes, encoder, pca


def build_mlp_model(input_shape, num_classes, unidades, learning_rate, dropout_rate):
    """
    Construye y compila un modelo de Keras tipo Perceptrón Multicapa (MLP).
    Incorpora BatchNormalization profundo y Dropout estable.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.GaussianNoise(0.1))
    
    # BatchNormalization a la entrada para estabilizar el PCA
    model.add(tf.keras.layers.BatchNormalization())
    
    for units in unidades:
        model.add(tf.keras.layers.Dense(units))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('swish'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        
    # Capa de salida proporcional al número de clases utilizando softmax
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def entrenar_modelo_parametrizado(X_train, X_val, X_test, y_train, y_val, y_test, num_classes):
    """
    Entrena el modelo MLP con parámetros específicos y sube el modelo resultante a wandb.
    """
    os.environ["WANDB_MODE"] = "online"
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)

    unidades = list(ARQUITECTURA)
    nombre_arq = "-".join(map(str, unidades))
    
    input_shape = (X_train.shape[1],)
    
    # Inicializar wandb
    run = wandb.init(
        entity="pd1-c2526-team2",
        project="clasificador-imagenes",
        name=f"mlp_parametrizado-{nombre_arq}-lr-{LEARNING_RATE_INIT}",
        job_type="train",
        config={
            "algoritmo": "MLP_PCA",
            "arquitectura": unidades,
            "learning_rate": LEARNING_RATE_INIT,
            "dropout": DROPOUT,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "embedding_source": ARCHIVO_EMBEDDINGS
        }
    )
    
    # Definir métricas en W&B
    wandb.define_metric("epoch")
    wandb.define_metric("epoch/*", step_metric="epoch")
    
    modelo = build_mlp_model(input_shape, num_classes, unidades, LEARNING_RATE_INIT, DROPOUT)
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3
    )
    wandb_metrics = wandb.keras.WandbMetricsLogger()
    
    print(f"\nEntrenando modelo con arquitectura {nombre_arq}, LR {LEARNING_RATE_INIT}, Dropout {DROPOUT}...")
    
    # Entrenar modelo
    history = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr, wandb_metrics],
        verbose=1
    )
    
    # Realizar predicciones finales post-entrenamiento (test set)
    y_pred_probs = modelo.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    test_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    
    print(f"\nResultados Test -> Test Acc: {test_acc:.3f} | F1: {f1:.3f} | Recall: {recall:.3f}")
    
    wandb.log({
        "test_final_accuracy": test_acc,
        "test_final_f1_score": f1,
        "test_final_recall": recall
    })
    
    # Guardar modelo
    nombre_guardado = "best_mlp_parametrizado.keras"
    print(f"\nGuardando el mejor modelo en {nombre_guardado}...")
    modelo.save(nombre_guardado)
    
    # Subir modelo a WandB como artefacto
    print(f"Subiendo artefacto del modelo {nombre_guardado} a W&B...")
    artifact = wandb.Artifact(name=f"mlp_model_{nombre_arq}_lr{LEARNING_RATE_INIT}", type='model')
    artifact.add_file(nombre_guardado)
    run.log_artifact(artifact)
    
    print("Artefacto subido correctamente.")
    run.finish()
    
    return modelo


if __name__ == '__main__':
    print(f"Conectando a MinIO y descargando datos ({ARCHIVO_EMBEDDINGS})...")
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes, encoder, pca_model = cargar_y_preparar_datos(ARCHIVO_EMBEDDINGS)
    
    print(f"Estructura de entrenamiento tras PCA: {X_train.shape}")
    print(f"Clases detectadas: {encoder.classes_} ({num_classes})")
    
    modelo = entrenar_modelo_parametrizado(X_train, X_val, X_test, y_train, y_val, y_test, num_classes)
    
    print("\nGuardando PCA Label Encoder y modelo PCA...")
    joblib.dump(encoder, "label_encoder_parametrizado.pkl")
    joblib.dump(pca_model, "pca_model_parametrizado.pkl")
    print("¡Proceso finalizado con éxito!")
