import numpy as np
import pandas as pd
import wandb
import optuna
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.base import clone

import nltk
from nltk.corpus import stopwords

from funciones_texto import bajar_df_texto, x_y_split


# =========================
# CONFIG WANDB
# =========================
WANDB_ENTITY = "pd1-c2526-team2"
WANDB_PROJECT = "modelo-texto-final"
WANDB_GROUP = "texto"
MODEL_TYPE = "svm"
RANDOM_STATE = 42
CV_N_SPLITS = 5

ARTIFACT_NAME = "best-svm-texto"
MODEL_FILENAME = "best_svm_texto.joblib"


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
# ARTIFACT W&B
# =========================
def guardar_artifact(model, metadata, run):
    joblib.dump(model, MODEL_FILENAME)

    artifact = wandb.Artifact(
        name=ARTIFACT_NAME,
        type="model",
        metadata=metadata,
        description="Mejor SVM (LinearSVC + TFIDF)"
    )

    artifact.add_file(MODEL_FILENAME)
    run.log_artifact(artifact)


# =========================
# GRID SEARCH
# =========================
def entrenar_svm_texto(X_train, y_train, X_test, y_test):

    stop_words = stopwords.words("spanish")
    skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name="svm-grid",
        job_type="grid-search",
        config={"model": MODEL_TYPE, "search": "grid"}
    )

    Cs = [0.1, 1.0, 10.0]
    ngrams = [(1, 1), (1, 2)]
    max_features_list = [5000, 10000]

    best_score = -1
    best_model = None
    best_params = None

    print("Buscando mejor SVM (Grid)...")

    for C in Cs:
        for ngram in ngrams:
            for max_feat in max_features_list:

                model = Pipeline([
                    ("tfidf", TfidfVectorizer(
                        max_features=max_feat,
                        ngram_range=ngram,
                        stop_words=stop_words
                    )),
                    ("clf", LinearSVC(C=C, class_weight="balanced", random_state=RANDOM_STATE))
                ])

                scores = []

                for tr_idx, val_idx in skf.split(X_train, y_train):
                    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

                    m = clone(model)
                    m.fit(X_tr, y_tr)
                    scores.append(f1_score(y_val, m.predict(X_val), average="macro"))

                mean_f1 = np.mean(scores)

                wandb.log({
                    "cv_f1": mean_f1,
                    "C": C,
                    "ngram": str(ngram),
                    "max_features": max_feat
                })

                if mean_f1 > best_score:
                    best_score = mean_f1
                    best_model = model.fit(X_train, y_train)
                    best_params = {
                        "C": C,
                        "ngram": ngram,
                        "max_features": max_feat
                    }

    print("\nBEST GRID:", best_score)

    metrics = evaluar_modelo(best_model, X_test, y_test)

    wandb.log({
        "test_metrics": metrics,
        "best_cv_f1": best_score
    })

    guardar_artifact(best_model, best_params, run)

    run.finish()
    return best_model, best_params


# =========================
# OPTUNA
# =========================
def entrenar_svm_texto_optuna(X_train, y_train, X_test, y_test):

    stop_words = stopwords.words("spanish")
    skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name="svm-optuna",
        job_type="optuna",
        config={"model": MODEL_TYPE, "search": "optuna"}
    )

    def objective(trial):

        C = trial.suggest_float("C", 0.01, 10.0, log=True)
        ngram = trial.suggest_categorical("ngram", [(1, 1), (1, 2)])
        max_features = trial.suggest_int("max_features", 2000, 20000, step=2000)

        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram,
                stop_words=stop_words
            )),
            ("clf", LinearSVC(C=C, class_weight="balanced", random_state=RANDOM_STATE))
        ])

        scores = []

        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            m = clone(model)
            m.fit(X_tr, y_tr)
            scores.append(f1_score(y_val, m.predict(X_val), average="macro"))

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    best_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=best_params["max_features"],
            ngram_range=best_params["ngram"],
            stop_words=stop_words
        )),
        ("clf", LinearSVC(
            C=best_params["C"],
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    best_model.fit(X_train, y_train)

    metrics = evaluar_modelo(best_model, X_test, y_test)

    wandb.log({
        "test_metrics": metrics,
        "best_cv_f1": study.best_value
    })

    guardar_artifact(best_model, best_params, run)

    run.finish()
    return best_model, best_params


# =========================
# MAIN
# =========================
def main():

    nltk.download("stopwords", quiet=True)
    wandb.login()

    df = bajar_df_texto()
    X, y = x_y_split(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    modo = input("1 (Grid) / 2 (Optuna): ")

    if modo == "1":
        entrenar_svm_texto(X_train, y_train, X_test, y_test)
    else:
        entrenar_svm_texto_optuna(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()