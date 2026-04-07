import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import wandb

from utils.funciones_minio import crear_cliente_minio, bajar_minio
from utils.config import PATH_DATASETS_MODELOS
from funciones_texto import bajar_df_texto, x_y_split, train_val_test_split

def entrenar_evaluar_naive_bayes(x_train, y_train, x_val, y_val):
    """
    Entrena un modelo Naive Bayes usando TF-IDF, evalúa el resultado
    sobre el conjunto de prueba y sube las métricas a W&B.
    """
    print("Descargando stopwords y configurando modelo...")
    nltk.download('stopwords')
    spanish_stopwords = stopwords.words('spanish')

    # 1. Definir el pipeline con TF-IDF y Multinomial Naive Bayes
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,2),
            stop_words=spanish_stopwords
        )),
        ("clf", MultinomialNB())
    ])

    # 2. Entrenar el modelo
    print("Entrenando modelo Naive Bayes...")
    model.fit(x_train, y_train)

    # 3. Generar predicciones para el conjunto de prueba
    y_pred = model.predict(x_val)

    # 4. Calcular métrica principal (Accuracy)
    accuracy_nb = accuracy_score(y_val, y_pred)

    # 5. Iniciar ejecución y registrar métricas en W&B
    run = wandb.init(
        entity="pd1-c2526-team2",
        project="modelo-texto",
        name="naive-bayes-tfidf",
        job_type="model-evaluation",
        group="texto_modelos",
    )

    wandb.log({
        "val_accuracy": accuracy_nb,
        "modelo": "Naive Bayes (TF-IDF)"
    })

    run.finish()

    # 6. Generar reportes adicionales para la consola
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)
    
    print("\nReporte de Clasificación:")
    print(report)

    return model, accuracy_nb, report, matrix

if __name__ == "__main__":
    # Descarga y partición de datos
    df = bajar_df_texto()
    x, y = x_y_split(df)
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y)
    
    # Ejecutar la función encapsulada
    entrenar_evaluar_naive_bayes(x_train, y_train, x_val, y_val)