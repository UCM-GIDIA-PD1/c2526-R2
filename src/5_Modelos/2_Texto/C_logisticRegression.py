import wandb
import numpy as np
import random

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import nltk
from nltk.corpus import stopwords

from funciones_texto import bajar_df_texto, x_y_split, train_val_test_split


def evaluar_modelo(model, X_val, y_val):
    """Evalúa un modelo de clasificación sobre un conjunto de validación y devuelve las métricas de evaluación seleccionadas

    Args:
        model: Modelo entrenado con método "predict".
        X_val: Datos de entrada del conjunto de validación.
        y_val: Etiquetas reales del conjunto de validación.

    Returns:
        dict: Diccionario con las métricas calculadas:
            - accuracy (float)
            - f1_macro (float)
            - recall_macro (float)
            - precision_macro (float)
    """
    
    y_pred = model.predict(X_val)

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1_macro": f1_score(y_val, y_pred, average="macro"),
        "recall_macro": recall_score(y_val, y_pred, average="macro"),
        "precision_macro": precision_score(y_val, y_pred, average="macro"),
    }


def entrenar_logreg_texto(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrena y selecciona el mejor modelo de clasificación de texto basado en regresión logística.
    Realiza una búsqueda exhaustiva sobre distintas combinaciones de:
    - Vectorizadores (CountVectorizer y TfidfVectorizer)
    - Valores de regularización (C)
    - Rangos de n-gramas
    - Número máximo de características

    Evalúa cada configuración en el conjunto de validación usando F1 macro
    como métrica principal, registra los resultados en Weights & Biases (wandb)
    y selecciona el mejor modelo para evaluarlo según el test.

    Args:
        X_train: Textos de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        X_val: Textos de validación.
        y_val: Etiquetas de validación.
        X_test: Textos de test.
        y_test: Etiquetas de test.

    Returns:
        tuple:
            - mejor_modelo: Pipeline entrenado con la mejor configuración encontrada.
            - mejor_resultado (dict): Diccionario con la configuración y métricas del mejor modelo.
    """
    spanish_stopwords = stopwords.words("spanish")

    vectorizers = {
        "count": CountVectorizer,
        "tfidf": TfidfVectorizer
    }

    Cs = [0.1, 1.0, 5.0]
    ngrams = [(1,1), (1,2)]
    max_features_list = [5000, 10000]

    mejor_resultado = None
    mejor_modelo = None

    run = wandb.init(
        entity="pd1-c2526-team2",
        project="modelo-texto-logreg",
        name="logreg-texto",
        job_type="model-training",
        config={
            "modelo": "LogisticRegression",
            "task": "clasificacion",
            "vectorizers": ["count", "tfidf"],
            "C_values": Cs,
            "ngram_range": ngrams,
            "max_features": max_features_list,
            "split": "train/val/test",
            "random_state": 42
        }
    )
    
    print("Buscando mejor modelo de texto...")
    table = wandb.Table(columns=[
        "modelo", "vectorizer", "C", "ngram", "max_features",
        "f1_macro", "accuracy", "recall_macro", "precision_macro"
    ])
    for vec_name, vec_class in vectorizers.items():
        for C in Cs:
            for ngram in ngrams:
                for max_feat in max_features_list:

                    nombre = f"{vec_name}_C{C}_ng{ngram}_mf{max_feat}"
                    print(f"\nEntrenando: {nombre}")

                    model = Pipeline([
                        ("vectorizer", vec_class(
                            max_features=max_feat,
                            ngram_range=ngram,
                            stop_words=spanish_stopwords
                        )),
                        ("classifier", LogisticRegression(
                            C=C,
                            max_iter=1000,
                            class_weight="balanced"
                        ))
                    ])

                    model.fit(X_train, y_train)
                    metricas = evaluar_modelo(model, X_val, y_val)

                    print(metricas)

                    table.add_data(
                        nombre,
                        vec_name,
                        C,
                        str(ngram),
                        max_feat,
                        metricas["f1_macro"],
                        metricas["accuracy"],
                        metricas["recall_macro"],
                        metricas["precision_macro"]
                    )

                    if (mejor_resultado is None) or (metricas["f1_macro"] > mejor_resultado["f1_macro"]):
                        mejor_resultado = metricas | {
                            "nombre": nombre,
                            "vectorizer": vec_name,
                            "C": C,
                            "ngram": ngram,
                            "max_features": max_feat
                        }
                        mejor_modelo = model
        

    print("\n=== MEJOR MODELO ===")
    print(mejor_resultado)

    wandb.log({"resultados_modelos": table})

    wandb.log({
    "f1_por_modelo": wandb.plot.bar(
        table,
        "modelo",
        "f1_macro",
        title="F1 por modelo"
    )
})
    
    metricas_test = evaluar_modelo(mejor_modelo, X_test, y_test)

    print("\n=== RESULTADOS EN TEST ===")
    print(metricas_test)

    wandb.log({
        "test_f1": metricas_test["f1_macro"],
        "test_accuracy": metricas_test["accuracy"],
        "test_recall": metricas_test["recall_macro"],
        "test_precision": metricas_test["precision_macro"]
    })

    run.finish()

    return mejor_modelo, mejor_resultado


def main():
    """Función principal del script.

    Descarga los recursos necesarios de NLTK, carga y prepara los datos de texto,
    realiza la partición en conjuntos de entrenamiento, validación y test, inicia
    sesión en Weights & Biases y lanza el proceso de entrenamiento y evaluación
    del modelo de clasificación de texto.
    """
    nltk.download("stopwords")

    df = bajar_df_texto()
    X, y = x_y_split(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    wandb.login()

    entrenar_logreg_texto(X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    main()