import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import wandb

from utils.funciones_minio import crear_cliente_minio, bajar_minio
from utils.config import PATH_DATASETS_MODELOS
from funciones_texto import bajar_df_texto, x_y_split, train_val_test_split

def entrenar_evaluar_svm(X_train, y_train, X_val, y_val):
    print("Descargando stopwords y configurando modelo...")
    nltk.download('stopwords', quiet=True)
    spanish_stopwords = stopwords.words('spanish')

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,2),
            stop_words=spanish_stopwords
        )),
        ("clf", LinearSVC(random_state=42))
    ])

    print("Entrenando modelo SVM...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    accuracy_svm = accuracy_score(y_val, y_pred)

    run = wandb.init(
        entity="pd1-c2526-team2",
        project="modelo-texto",
        name="svm-tfidf",
        job_type="model-evaluation",
        group="texto_modelos",
    )

    wandb.log({
        "val_accuracy": accuracy_svm,
        "modelo": "SVM (TF-IDF)"
    })

    run.finish()

    report = classification_report(y_val, y_pred, zero_division=0)
    matrix = confusion_matrix(y_val, y_pred)
    
    print("\nReporte de Clasificación:")
    print(report)

    return model, accuracy_svm, report, matrix

if __name__ == "__main__":
    print("Descargando y preparando los datos...")
    df = bajar_df_texto()
    x, y = x_y_split(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(x, y)
    print("Datos descargados.")
    entrenar_evaluar_svm(X_train, y_train, X_val, y_val)