import wandb
import numpy as np
import random

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import optuna

import nltk
from nltk.corpus import stopwords


from funciones_texto import bajar_df_texto, x_y_split, train_val_test_split


def evaluar_modelo(model, X_test, y_test):
    """Evalúa un modelo de clasificación sobre un conjunto de validación y devuelve las métricas de evaluación seleccionadas

    Args:
        model: Modelo entrenado con método "predict".
        X_test: Datos de entrada del conjunto de validación.
        y_test: Etiquetas reales del conjunto de validación.

    Returns:
        dict: Diccionario con las métricas calculadas:
            - accuracy (float)
            - f1_macro (float)
            - recall_macro (float)
            - precision_macro (float)
    """
    
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
    }

def analizar_por_longitud(model, X_test, y_test):
    import pandas as pd

    df = pd.DataFrame({
        "texto": X_test,
        "y_true": y_test
    })

    df["longitud"] = df["texto"].apply(lambda x: len(x.split()))

    # Definir segmentos
    bins = [0, 20, 50, 100, np.inf]
    labels = ["corto", "medio", "largo", "muy_largo"]
    df["segmento"] = pd.cut(df["longitud"], bins=bins, labels=labels)

    resultados = []

    for seg in labels:
        subset = df[df["segmento"] == seg]
        if len(subset) == 0:
            continue

        y_pred = model.predict(subset["texto"])

        f1 = f1_score(subset["y_true"], y_pred, average="macro")

        resultados.append({
            "segmento": seg,
            "n_samples": len(subset),
            "f1_macro": f1
        })

    return pd.DataFrame(resultados)


def entrenar_logreg_texto(X_train, y_train, X_test, y_test):
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

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    """ run = wandb.init(
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
    ) """
    
    print("Buscando mejor modelo de texto...")
    """table = wandb.Table(columns=[
        "modelo", "vectorizer", "C", "ngram", "max_features",
        "f1_macro", "accuracy", "recall_macro", "precision_macro"
    ])"""
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
                            class_weight="balanced",
                            random_state=42
                        ))
                    ])

                    cv_scores = []

                    for train_idx, val_idx in skf.split(X_train, y_train):
                        
                        X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]

                        model_cv = clone(model)
                        model_cv.fit(X_tr, y_tr)

                        y_pred = model_cv.predict(X_vl)
                        f1 = f1_score(y_vl, y_pred, average="macro")

                        cv_scores.append(f1)

                    f1_cv_mean = np.mean(cv_scores)
                    f1_cv_std = np.std(cv_scores)

                    print(f1_cv_mean)
                    print(f1_cv_std)


                    """table.add_data(
                        nombre,
                        vec_name,
                        C,
                        str(ngram),
                        max_feat,
                        metricas["f1_macro"],
                        metricas["accuracy"],
                        metricas["recall_macro"],
                        metricas["precision_macro"]
                    )"""

                    if (mejor_resultado is None) or (f1_cv_mean > mejor_resultado["f1_macro"]):
                        mejor_resultado = {
                            "nombre": nombre,
                            "vectorizer": vec_name,
                            "C": C,
                            "ngram": ngram,
                            "max_features": max_feat,
                            "f1_macro": f1_cv_mean,
                            "f1_std": f1_cv_std
                        }
                        mejor_config = {
                            "vec_class": vec_class,
                            "C": C,
                            "ngram": ngram,
                            "max_features": max_feat
                        }

                        mejor_modelo = model
        

    print("\n=== MEJOR MODELO ===")
    print(mejor_resultado)
    mejor_modelo.fit(X_train, y_train)

    mejor_modelo = Pipeline([
        ("vectorizer", mejor_config["vec_class"](
            max_features=mejor_config["max_features"],
            ngram_range=mejor_config["ngram"],
            stop_words=spanish_stopwords
        )),
        ("classifier", LogisticRegression(
            C=mejor_config["C"],
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    mejor_modelo.fit(X_train, y_train)

    metricas_test = evaluar_modelo(mejor_modelo, X_test, y_test)

    #wandb.log({"resultados_modelos": table})

    """wandb.log({
    "f1_por_modelo": wandb.plot.bar(
        table,
        "modelo",
        "f1_macro",
        title="F1 por modelo"
    )})"""

    print("\n=== RESULTADOS EN TEST ===")
    print(metricas_test)

    """wandb.log({
        "test_f1": metricas_test["f1_macro"],
        "test_accuracy": metricas_test["accuracy"],
        "test_recall": metricas_test["recall_macro"],
        "test_precision": metricas_test["precision_macro"]
    })

    run.finish()"""


    print("\n=== ANALISIS POR CLASE ===")
    print(classification_report(y_test, mejor_modelo.predict(X_test)))

    print("\n=== ANALISIS POR LONGITUD ===")
    print(analizar_por_longitud(mejor_modelo, X_test, y_test))

    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, mejor_modelo.predict(X_test))

    print("\n=== MATRIZ DE CONFUSION ===")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    return mejor_modelo, mejor_resultado

def entrenar_logreg_texto_optuna(X_train, y_train, X_test, y_test):

    spanish_stopwords = stopwords.words("spanish")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):

        vec_name = trial.suggest_categorical("vectorizer", ["count", "tfidf"])
        C = trial.suggest_float("C", 0.1, 5.0)
        ngram = trial.suggest_categorical("ngram", [(1,1), (1,2)])
        max_features = trial.suggest_int("max_features", 1000, 15000, step=1000)

        vec_class = CountVectorizer if vec_name == "count" else TfidfVectorizer

        model = Pipeline([
            ("vectorizer", vec_class(
                max_features=max_features,
                ngram_range=ngram,
                stop_words=spanish_stopwords
            )),
            ("classifier", LogisticRegression(
                C=C,
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ])

        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):

            X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model_cv = clone(model)
            model_cv.fit(X_tr, y_tr)

            y_pred = model_cv.predict(X_vl)
            f1 = f1_score(y_vl, y_pred, average="macro")

            scores.append(f1)

            # 🔥 PRUNING
            trial.report(np.mean(scores), step=fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    print("Buscando mejor modelo con Optuna...")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )
    study.optimize(objective, n_trials=20)  # puedes subir esto

    best_params = study.best_params

    print("\n=== MEJOR CONFIG OPTUNA ===")
    print(best_params)

    vec_class = CountVectorizer if best_params["vectorizer"] == "count" else TfidfVectorizer

    mejor_modelo = Pipeline([
        ("vectorizer", vec_class(
            max_features=best_params["max_features"],
            ngram_range=best_params["ngram"],
            stop_words=spanish_stopwords
        )),
        ("classifier", LogisticRegression(
            C=best_params["C"],
            max_iter=1000,
            class_weight="balanced",
            random_state = 42
        ))
    ])

    mejor_modelo.fit(X_train, y_train)

    metricas_test = evaluar_modelo(mejor_modelo, X_test, y_test)

    print("\n=== RESULTADOS EN TEST ===")
    print(metricas_test)

    print("\n=== ANALISIS POR CLASE ===")
    print(classification_report(y_test, mejor_modelo.predict(X_test)))

    print("\n=== ANALISIS POR LONGITUD ===")
    print(analizar_por_longitud(mejor_modelo, X_test, y_test))

    cm = confusion_matrix(y_test, mejor_modelo.predict(X_test))

    print("\n=== MATRIZ DE CONFUSION ===")
    print(pd.DataFrame(cm))

    return mejor_modelo, best_params

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    """wandb.login()"""

    modo = input("Selecciona modo: 1 (Grid Search) / 2 (Optuna): ")

    if modo == "1":
        entrenar_logreg_texto(X_train, y_train, X_test, y_test)
    elif modo == "2":
        entrenar_logreg_texto_optuna(X_train, y_train, X_test, y_test)
    else:
        print("Opción no válida")
   


if __name__ == "__main__":
    main()