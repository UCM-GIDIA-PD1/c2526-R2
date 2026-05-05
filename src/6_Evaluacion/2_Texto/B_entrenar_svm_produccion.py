import os
import numpy as np
import pandas as pd
import wandb
import joblib

import nltk
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report,
)

from funciones_texto import bajar_df_texto, x_y_split


# =========================
# CONFIG WANDB (consistente con E_SVM.py)
# =========================
WANDB_ENTITY = "pd1-c2526-team2"
WANDB_PROJECT = "modelo-texto-final"
WANDB_GROUP = "texto"
MODEL_TYPE = "svm"
RANDOM_STATE = 42
CV_N_SPLITS = 5

ARTIFACT_NAME = "svm-texto-produccion"
MODEL_FILENAME = "svm_texto_produccion.joblib"

# Mejores hiperparámetros (extraídos del artefacto ganador en W&B)
MEJORES_PARAMS_SVM = {
    "vectorizer": {
        "max_df": 0.9469017639780448,
        "min_df": 4,
        "max_features": 15000,
        "ngram_range": (1, 2),
    },
    "classifier": {
        "C": 1.5532451953718347,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    },
}


# =========================
# PIPELINE
# =========================
def construir_pipeline(stop_words):
    p_vec = MEJORES_PARAMS_SVM["vectorizer"]
    p_clf = MEJORES_PARAMS_SVM["classifier"]
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_df=p_vec["max_df"],
            min_df=p_vec["min_df"],
            max_features=p_vec["max_features"],
            ngram_range=p_vec["ngram_range"],
            stop_words=stop_words,
        )),
        ("clf", LinearSVC(
            C=p_clf["C"],
            class_weight=p_clf["class_weight"],
            random_state=p_clf["random_state"],
        )),
    ])


# =========================
# ARTIFACT W&B
# =========================
def guardar_artifact(model, metadata, run):
    joblib.dump(model, MODEL_FILENAME)

    artifact = wandb.Artifact(
        name=ARTIFACT_NAME,
        type="model",
        metadata=metadata,
        description="SVM (TF-IDF + LinearSVC) reentrenado con el 100% de los datos para producción.",
    )
    artifact.add_file(MODEL_FILENAME)
    run.log_artifact(artifact)


# =========================
# ENTRENAMIENTO PRODUCCIÓN
# =========================
def entrenar_y_guardar_produccion(X, y):
    print("\nENTRENANDO MODELO DE PRODUCCIÓN SVM (CLASIFICACIÓN DE TEXTO)")
    print(f"Distribución de clases:\n{y.value_counts()}")

    stop_words = stopwords.words("spanish")

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name="svm-produccion-final",
        job_type="model-registry",
        config={
            "model": MODEL_TYPE,
            **MEJORES_PARAMS_SVM["vectorizer"],
            **MEJORES_PARAMS_SVM["classifier"],
        },
    )

    # 1. Partición 80/20 estratificada por clase
    print("Partición 80% entrenamiento / 20% test, estratificada por clase...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # 2. CROSS VALIDATION sobre el 80% (F1 macro)
    print(f"Aplicando {CV_N_SPLITS}-Fold Cross Validation sobre training...")
    skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        construir_pipeline(stop_words),
        X_train, y_train,
        cv=skf, scoring="f1_macro", n_jobs=-1,
    )
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    print(f"-> [CV {CV_N_SPLITS}-Folds] F1 macro: {cv_mean*100:.2f}% (±{cv_std*100:.2f})")

    # 3. EVALUACIÓN EN HOLD-OUT (20%)
    print("Entrenando temporalmente para hold-out...")
    modelo_holdout = construir_pipeline(stop_words)
    modelo_holdout.fit(X_train, y_train)
    y_pred = modelo_holdout.predict(X_test)

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
    }
    print("[Test] Métricas hold-out:")
    for k, v in test_metrics.items():
        print(f"   · {k:<18}: {v*100:.2f} %")
    print("\nInforme por clase en hold-out:")
    print(classification_report(y_test, y_pred, digits=4))

    # 4. ENTRENAMIENTO DEFINITIVO con el 100% (train + val + test)
    print("Entrenando el modelo definitivo con el 100% de los datos para producción...")
    modelo_produccion = construir_pipeline(stop_words)
    modelo_produccion.fit(X, y)

    # 5. LOG MÉTRICAS y SUBIDA DEL ARTEFACTO
    wandb.log({
        "cv_f1_macro_mean": cv_mean,
        "cv_f1_macro_std": cv_std,
        "test_metrics": test_metrics,
    })

    metadata = {
        **MEJORES_PARAMS_SVM["vectorizer"],
        **MEJORES_PARAMS_SVM["classifier"],
        "ngram_range": str(MEJORES_PARAMS_SVM["vectorizer"]["ngram_range"]),
        "cv_f1_macro_mean": cv_mean,
        "test_f1_macro": test_metrics["f1_macro"],
        "n_train_total": len(X),
    }

    try:
        print("Subiendo el modelo al Model Registry de W&B...")
        guardar_artifact(modelo_produccion, metadata, run)
        print("¡Artefacto subido con éxito a W&B!")
    finally:
        if os.path.exists(MODEL_FILENAME):
            os.remove(MODEL_FILENAME)
            print(f"El archivo temporal '{MODEL_FILENAME}' ha sido eliminado de tu equipo.")

    run.finish()
    return modelo_produccion


# =========================
# MAIN
# =========================
def main():
    nltk.download("stopwords", quiet=True)
    wandb.login()

    df = bajar_df_texto()
    X, y = x_y_split(df)

    entrenar_y_guardar_produccion(X, y)


if __name__ == "__main__":
    main()