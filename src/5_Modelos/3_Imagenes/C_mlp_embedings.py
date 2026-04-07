import os
import joblib
import numpy as np
import tensorflow as tf
import wandb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from utils.funciones_minio import crear_cliente_minio, bajar_minio
from utils.config import MINIO_EMBEDDINGS

CLASES_PERMITIDAS = ['Cocina', 'Dormitorio', 'Salón', 'Banyo']
EMBEDDINGS_IMAGENES = "embeddings_imagenes.parquet"
EMBEDDINGS_IMAGENES_PROPIA = "embeddings_cnn_propia.parquet"

def cargar_y_preparar_datos(archivo_embeddings=EMBEDDINGS_IMAGENES_PROPIA):
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


def entrenar_mlp(X_train, X_val, X_test, y_train, y_val, y_test, num_classes, prefijo_wandb="mlp_propia"):
    """
    Entrena modelos MLP variando hiperparámetros (arquitectura y learning rate).
    Guarda el mejor usando el VALIDATION SET y previene Memory Leaks.
    """
    arquitecturas = [
        [512, 256, 128],
        [1024, 512],
        [256, 64],
        [512, 128]
    ]
    learning_rates = [0.01, 0.001, 0.0005, 0.0001]
    
    dropout_rate = 0.3
    epochs = 40
    batch_size = 128
    
    input_shape = (X_train.shape[1],)
    
    mejor_val_accuracy = 0.0
    temp_best_model_path = "temp_best_mlp.keras"

    for unidades in arquitecturas:
        nombre_arq = "-".join(map(str, unidades))
        print(f"\n Iniciando pruebas para MLP con arquitectura='{nombre_arq}'...")
        
        for lr in tqdm(learning_rates, desc=f"Entrenando Arq ({nombre_arq}) con LRs"):
            run = wandb.init(
                entity="pd1-c2526-team2",
                project="clasificador-imagenes",
                name=f"{prefijo_wandb}-{nombre_arq}-lr-{lr}",
                job_type="hyperparameter-tuning",
                config={
                    "algoritmo": "MLP_PCA",
                    "arquitectura": unidades,
                    "learning_rate": lr,
                    "dropout": dropout_rate,
                    "epochs": epochs,
                    "batch_size": batch_size
                }
            )
            
            # Definir métricas en W&B para evitar desincronización
            wandb.define_metric("epoch")
            wandb.define_metric("epoch/*", step_metric="epoch")
            
            # 1. Construir modelo 
            modelo = build_mlp_model(input_shape, num_classes, unidades, lr, dropout_rate)
            
            # EarlyStopping monitoreando 'val_loss' (ahora sí validación, no test)
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True
            )
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=3
            )
            
            wandb_metrics = wandb.keras.WandbMetricsLogger()
            
            # 2. Entrenar modelo usando validación explícita estratificada
            history = modelo.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, reduce_lr, wandb_metrics],
                verbose=0
            )
            
            # 3. Realizar predicciones finales post-entrenamiento (test set)
            y_pred_probs = modelo.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = y_test
            
            # 4. Cálculo de métricas sobre test
            test_acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            
            # Obtener el validation accuracy del mejor epoch restaurado
            best_epoch_idx = np.argmin(history.history['val_loss'])
            val_acc = float(history.history['val_accuracy'][best_epoch_idx])
            
            print(f"   LR={lr:.4f} | Val Acc: {val_acc:.3f} | Test Acc: {test_acc:.3f} | F1: {f1:.3f}")
            
            wandb.log({
                "test_final_accuracy": test_acc,
                "test_final_f1_score": f1,
                "test_final_recall": recall
            })
            
            # Seleccionar hiperparámetros basándose en Validation, NO en Test Set
            if val_acc > mejor_val_accuracy:
                mejor_val_accuracy = val_acc
                modelo.save(temp_best_model_path)
                
                # Subimos el modelo a W&B como artefacto solo si es el nuevo "mejor"
                best_artifact = wandb.Artifact(name=f"mejor_modelo_{prefijo_wandb}", type="model")
                best_artifact.add_file(temp_best_model_path, name=f"best_mlp_{nombre_arq}_lr{lr}.keras")
                run.log_artifact(best_artifact)
            
            run.finish()
            
            # Liberar memoria de Keras para prevenir Memory Leaks en el bucle
            tf.keras.backend.clear_session()
        
    # Cargar el mejor modelo y limpiar el temporal
    if os.path.exists(temp_best_model_path):
        mejor_modelo_final = tf.keras.models.load_model(temp_best_model_path)
        os.remove(temp_best_model_path)
    else:
        mejor_modelo_final = None
        
    return mejor_modelo_final


if __name__ == '__main__':
    # Habilitamos ejecución de sincronización con Wandb en los mismos términos que los scripts base
    os.environ["WANDB_MODE"] = "online"
    
    # Fijar semillas globales para reproducibilidad
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)
    
    print("======== MENÚ DE EMBEDDINGS ========")
    print("1. embeddings_imagenes.parquet (Dataset original)")
    print("2. embeddings_cnn_propia.parquet (Modelo CNN propio)")
    opcion = input("Seleccione el archivo a utilizar (1 o 2) [por defecto 2]: ").strip()
    
    if opcion == "1":
        archivo_seleccionado = EMBEDDINGS_IMAGENES
        prefijo_wandb = "mlp_base"
        nombre_guardado = "best_mlp_model_cnn_base.keras"
    else:
        archivo_seleccionado = EMBEDDINGS_IMAGENES_PROPIA
        prefijo_wandb = "mlp_propia"
        nombre_guardado = "best_mlp_model_cnn_propio.keras"
        
    print(f"\nConectando a MinIO y descargando datos de '{archivo_seleccionado}'...")
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes, encoder, pca_model = cargar_y_preparar_datos(archivo_seleccionado)
    
    print(f"Estructura de entrenamiento tras PCA: {X_train.shape}")
    print(f"Clases detectadas: {encoder.classes_} ({num_classes})")
    
    print("Iniciando fase experimental del MLP...")
    # Entrenar multi-layer perceptron iterando hiperparámetros
    mejor_mlp = entrenar_mlp(X_train, X_val, X_test, y_train, y_val, y_test, num_classes, prefijo_wandb)
    
    print(f"\nGuardando el mejor modelo en '{nombre_guardado}', junto al codificador y transformador PCA...")
    mejor_mlp.save(nombre_guardado)
    joblib.dump(encoder, "label_encoder.pkl")
    joblib.dump(pca_model, "pca_model.pkl") # Importante guardar el objeto PCA
    print("Artefactos guardados con éxito.")
    print("¡Proceso completado con éxito!")
