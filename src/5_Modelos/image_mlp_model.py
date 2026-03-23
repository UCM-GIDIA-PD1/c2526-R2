import os
import joblib
import numpy as np
import tensorflow as tf
import wandb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder

from utils.funciones_minio import crear_cliente_minio, bajar_minio
from utils.config import MINIO_EMBEDDINGS

CLASES_PERMITIDAS = ['Cocina', 'Dormitorio', 'Salón', 'Banyo']
EMBEDDINGS_IMAGENES = "embeddings_imagenes.parquet"

def cargar_y_preparar_datos():
    """
    Descarga los embeddings de MinIO, filtra por clases permitidas,
    codifica las etiquetas (one-hot para la red neuronal) y divide en train/test.
    """
    cliente = crear_cliente_minio()
    df = bajar_minio(cliente, MINIO_EMBEDDINGS, EMBEDDINGS_IMAGENES)
    
    # Filtrar solo por las clases permitidas
    df = df[df['clase'].isin(CLASES_PERMITIDAS)]
    
    # Extraer variables X e y
    X = np.stack(df["embedding"].values)
    y_raw = df["clase"].values
    
    # Codificar la variable objetivo (y) de string a numérico
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)
    
    # Convertir a matriz categórica tipo one-hot (necesario para la salida softmax)
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    
    # División de datos de forma estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.3, random_state=101, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, num_classes, encoder


def build_mlp_model(input_shape, num_classes, units_1, units_2, learning_rate, dropout_rate):
    """
    Construye y compila un modelo de Keras tipo Perceptrón Multicapa (MLP).
    Permite parametrizar las capas ocultas y el learning_rate para facilitar 
    la búsqueda de hiperparámetros.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # BatchNormalization ayuda a estabilizar la entrada de embeddings que puede variar mucho
        tf.keras.layers.BatchNormalization(),
        
        # Primera capa oculta
        tf.keras.layers.Dense(units_1, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate), 
        
        # Segunda capa oculta
        tf.keras.layers.Dense(units_2, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate / 2),
        
        # Capa de salida proporcional al número de clases utilizando softmax
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def entrenar_mlp(X_train, X_test, y_train, y_test, num_classes):
    """
    Entrena modelos MLP variando hiperparámetros (arquitectura y learning rate).
    Sube las ejecuciones y evalúa métricas en Weights & Biases en Runs separados 
    orientados a la búsqueda de hiperparámetros.
    Réplica de la lógica de experimentación vista en 'entrenar_knn'.
    """
    # Hiperparámetros de búsqueda definidos basados en las pruebas típicas de redes neuronales
    arquitecturas = [
        (512, 128),
        (256, 64)
    ]
    learning_rates = [0.001, 0.0005, 0.0001]
    
    # Parámetros estables propuestos
    dropout_rate = 0.3
    epochs = 40
    batch_size = 128
    
    input_shape = (X_train.shape[1],)
    
    # Variables auxiliares para preservar el mejor modelo global procesado
    mejor_modelo = None
    mejor_accuracy = 0.0
    
    # Muestra representativa para calcular el Train Accuracy de manera rápida
    tamano_muestra = min(5000, len(X_train))
    indices_aleatorios = np.random.choice(len(X_train), tamano_muestra, replace=False)
    X_train_muestra = X_train[indices_aleatorios]
    y_train_muestra = y_train[indices_aleatorios]

    for units_1, units_2 in arquitecturas:
        print(f"\n Iniciando pruebas para MLP con arquitectura='{units_1}-{units_2}'...")
        
        # Inicializamos un Run en W&B por cada configuración mayor (en este caso la arquitectura)
        run = wandb.init(
            entity="pd1-c2526-team2",
            project="clasificador-imagenes",
            name=f"mlp-arq-{units_1}-{units_2}",
            job_type="hyperparameter-tuning",
            config={
                "algoritmo": "MLP",
                "capa_1_units": units_1,
                "capa_2_units": units_2,
                "dropout": dropout_rate,
                "epochs": epochs,
                "batch_size": batch_size
            }
        )

        wandb.define_metric("learning_rate")
        wandb.define_metric("train_accuracy", step_metric="learning_rate")
        wandb.define_metric("accuracy", step_metric="learning_rate")
        wandb.define_metric("f1_score", step_metric="learning_rate")
        wandb.define_metric("recall", step_metric="learning_rate")
        
        for lr in tqdm(learning_rates, desc=f"Entrenando Arq ({units_1}-{units_2}) con distintos LRs"):
            
            # 1. Construir modelo 
            modelo = build_mlp_model(input_shape, num_classes, units_1, units_2, lr, dropout_rate)
            
            # Incorporamos EarlyStopping para evitar invertir tiempo en un modelo estancado
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True
            )
            
            # 2. Entrenar modelo
            # Utilizamos verbose=0 para no saturar el output y que se respete la visibilidad de tqdm
            modelo.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            # 3. Realizar predicciones
            y_pred_probs = modelo.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1) # Deshace el categórico temporal para validación
            y_true = np.argmax(y_test, axis=1)
            
            y_pred_train_probs = modelo.predict(X_train_muestra, verbose=0)
            y_pred_train = np.argmax(y_pred_train_probs, axis=1)
            y_true_train = np.argmax(y_train_muestra, axis=1)
            
            # 4. Cálculo de métricas
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            train_acc = accuracy_score(y_true_train, y_pred_train)
            
            print(f"   LR={lr:.4f} | Train Acc: {train_acc:.3f} | Test Acc: {acc:.3f} | F1: {f1:.3f} | Recall: {recall:.3f}")
            
            # 5. Log de información en WandB iterando sobre el parámetro hipervariable secundario
            wandb.log({
                "learning_rate": lr,  
                "train_accuracy": train_acc,
                "accuracy": acc,        
                "f1_score": f1,         
                "recall": recall
            })
            
            if acc > mejor_accuracy:
                mejor_accuracy = acc
                mejor_modelo = modelo
            
        run.finish()
        
    return mejor_modelo


if __name__ == '__main__':
    # Habilitamos ejecución de sincronización con Wandb en los mismos términos que los scripts base
    os.environ["WANDB_MODE"] = "online"
    
    print("Conectando a MinIO y descargando datos...")
    X_train, X_test, y_train, y_test, num_classes, encoder = cargar_y_preparar_datos()
    
    print(f"Estructura de entrenamiento: {X_train.shape}")
    print(f"Clases detectadas: {encoder.classes_} ({num_classes})")
    
    print("Iniciando fase experimental del MLP...")
    # Entrenar multi-layer perceptron iterando hiperparámetros y recogiendo el modelo superior
    mejor_mlp = entrenar_mlp(X_train, X_test, y_train, y_test, num_classes)
    
    # Opcional pero recomendando retener el mejor modelo para su futura inferencia
    print("\nGuardando el mejor modelo y el codificador...")
    mejor_mlp.save("best_mlp_model.keras")
    joblib.dump(encoder, "label_encoder.pkl")
    print("Artefactos guardados con éxito: 'best_mlp_model.keras', 'label_encoder.pkl'.")
    print("¡Proceso completado con éxito!")
