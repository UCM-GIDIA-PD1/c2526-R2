import numpy as np
import pandas as pd
import optuna
import wandb

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone

import nltk
from nltk.corpus import stopwords

from funciones_texto import bajar_df_texto, x_y_split


# =========================
# MÉTRICAS
# =========================
def evaluar_modelo(model, X_test, y_test):

    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
    }


# =========================
# GRID SEARCH + WANDB (1 RUN)
# =========================
def entrenar_logreg_texto(X_train, y_train, X_test, y_test):

    spanish_stopwords = stopwords.words("spanish")

    vectorizers = {
        "count": CountVectorizer,
        "tfidf": TfidfVectorizer
    }

    Cs = [0.1, 1.0, 5.0]
    ngrams = [(1, 1), (1, 2)]
    max_features_list = [5000, 10000]

    mejor_resultado = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    run = wandb.init(
        project="modelo-texto-f",
        group="logreg-grid",
        name="logreg-grid",
        tags=["logreg", "grid"],
        config={"model": "logreg", "search": "grid"}
    )

    print("Buscando mejor Logistic Regression (Grid)...")

    for vec_name, vec_class in vectorizers.items():
        for C in Cs:
            for ngram in ngrams:
                for max_feat in max_features_list:

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

                        X_tr = X_train.iloc[train_idx]
                        X_vl = X_train.iloc[val_idx]
                        y_tr = y_train.iloc[train_idx]
                        y_vl = y_train.iloc[val_idx]

                        model_cv = clone(model)
                        model_cv.fit(X_tr, y_tr)

                        y_pred = model_cv.predict(X_vl)
                        f1 = f1_score(y_vl, y_pred, average="macro")

                        cv_scores.append(f1)

                    f1_mean = np.mean(cv_scores)
                    f1_std = np.std(cv_scores)

                    wandb.log({
                        "cv_f1_mean": f1_mean,
                        "cv_f1_std": f1_std,
                        "C": C,
                        "ngram": str(ngram),
                        "max_features": max_feat,
                        "vectorizer": vec_name
                    })

                    if (mejor_resultado is None) or (f1_mean > mejor_resultado["f1_macro"]):
                        mejor_resultado = {
                            "vectorizer": vec_name,
                            "C": C,
                            "ngram": ngram,
                            "max_features": max_feat,
                            "f1_macro": f1_mean,
                            "f1_std": f1_std
                        }
                        mejor_config = {
                            "vec_class": vec_class,
                            "C": C,
                            "ngram": ngram,
                            "max_features": max_feat
                        }

    wandb.log({
        "best_cv_f1": mejor_resultado["f1_macro"],
        "best_params": mejor_resultado
    })

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

    wandb.log({
        "test_f1_macro": metricas_test["f1_macro"],
        "test_accuracy": metricas_test["accuracy"],
        "test_precision": metricas_test["precision_macro"],
        "test_recall": metricas_test["recall_macro"]
    })

    print("\n=== RESULTADOS EN TEST ===")
    print(metricas_test)

    run.finish()

    return mejor_modelo, mejor_resultado


# =========================
# OPTUNA + WANDB (1 RUN)
# =========================
def entrenar_logreg_texto_optuna(X_train, y_train, X_test, y_test):

    spanish_stopwords = stopwords.words("spanish")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    run = wandb.init(
        project="modelo-texto-f",
        group="logreg-optuna",
        name="logreg-optuna",
        tags=["logreg", "optuna"],
        config={"model": "logreg", "search": "optuna"}
    )

    def objective(trial):

        vec_name = trial.suggest_categorical("vectorizer", ["count", "tfidf"])
        C = trial.suggest_float("C", 0.1, 5.0)
        ngram = trial.suggest_categorical("ngram", [(1, 1), (1, 2)])
        max_features = trial.suggest_int("max_features", 2000, 15000, step=2000)

        vec_class = CountVectorizer if vec_name == "count" else TfidfVectorizer

        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):

            X_tr = X_train.iloc[train_idx]
            X_vl = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_vl = y_train.iloc[val_idx]

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

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_vl)

            f1 = f1_score(y_vl, y_pred, average="macro")
            scores.append(f1)

            trial.report(np.mean(scores), step=fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    wandb.log({"best_params": best_params})

    vec_class = CountVectorizer if best_params["vectorizer"] == "count" else TfidfVectorizer

    best_model = Pipeline([
        ("vectorizer", vec_class(
            max_features=best_params["max_features"],
            ngram_range=best_params["ngram"],
            stop_words=spanish_stopwords
        )),
        ("classifier", LogisticRegression(
            C=best_params["C"],
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    best_model.fit(X_train, y_train)

    metricas_test = evaluar_modelo(best_model, X_test, y_test)

    wandb.log({
        "test_f1_macro": metricas_test["f1_macro"],
        "test_accuracy": metricas_test["accuracy"],
        "test_precision": metricas_test["precision_macro"],
        "test_recall": metricas_test["recall_macro"]
    })

    print("\n=== RESULTADOS EN TEST ===")
    print(metricas_test)

    run.finish()

    return best_model, best_params


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    nltk.download("stopwords")

    df = bajar_df_texto()
    X, y = x_y_split(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    modo = input("Selecciona modo: 1 (Grid) / 2 (Optuna): ")

    if modo == "1":
        entrenar_logreg_texto(X_train, y_train, X_test, y_test)
    else:
        entrenar_logreg_texto_optuna(X_train, y_train, X_test, y_test)