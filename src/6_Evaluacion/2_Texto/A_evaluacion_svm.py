import numpy as np
import pandas as pd
import wandb
import joblib
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
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
# EVALUACIÓN
# =========================
def evaluar_svm_final(X_train, y_train, X_test, y_test):
    print("EVALUACIÓN FINAL SVM (CLASIFICACIÓN DE TEXTO)")

    stop_words = stopwords.words("spanish")

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name="svm-evaluacion-final",
        job_type="model-evaluation",
        config={
            "model": MODEL_TYPE,
            **MEJORES_PARAMS_SVM["vectorizer"],
            **MEJORES_PARAMS_SVM["classifier"],
        },
    )

    modelo_final = construir_pipeline(stop_words)

    print("Entrenando modelo definitivo sobre el 80%...")
    modelo_final.fit(X_train, y_train)

    y_pred = modelo_final.predict(X_test)

    # 1. MÉTRICAS GLOBALES
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
    }

    print("\nMÉTRICAS GLOBALES EN TEST:")
    for k, v in metrics.items():
        print(f"   · {k:<18}: {v*100:.2f} %")

    # 2. INFORME POR CLASE
    print("\nINFORME DETALLADO POR CLASE:")
    print(classification_report(y_test, y_pred, digits=4))

    # 3. MATRIZ DE CONFUSIÓN
    clases = modelo_final.named_steps["clf"].classes_
    cm = confusion_matrix(y_test, y_pred, labels=clases)

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases).plot(
        ax=ax, cmap="Blues", colorbar=True
    )
    ax.set_title("Matriz de Confusión - SVM Clasificación de Texto", fontsize=14)
    plt.tight_layout()

    # Guardamos en disco en lugar de plt.show() para evitar bloqueo en terminales sin GUI
    cm_path = "matriz_confusion_svm_texto.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Matriz de confusión guardada en: {cm_path}")

    # 4. TOP PALABRAS POR CLASE (aprovechamos que LinearSVC es lineal -> coef_)
    vectorizer = modelo_final.named_steps["tfidf"]
    classifier = modelo_final.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = classifier.coef_  # shape (n_clases, n_features)

    print("\nTOP 10 TÉRMINOS MÁS DISCRIMINANTES POR CLASE:")
    for i, clase in enumerate(clases):
        top_idx = np.argsort(coefs[i])[::-1][:10]
        print(f"\n   [{clase}]")
        for palabra, peso in zip(feature_names[top_idx], coefs[i][top_idx]):
            print(f"      · {palabra:<30} (peso: {peso:.3f})")

    # 5. LOG W&B
    # wandb.plot.confusion_matrix necesita índices enteros, no strings -> mapeamos
    clase_a_idx = {c: i for i, c in enumerate(clases)}
    y_test_idx = [clase_a_idx[c] for c in y_test]
    y_pred_idx = [clase_a_idx[c] for c in y_pred]

    wandb.log({
        "test_metrics": metrics,
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test_idx,
            preds=y_pred_idx,
            class_names=list(clases),
        ),
        "confusion_matrix_img": wandb.Image(cm_path),
    })

    run.finish()
    return modelo_final, metrics


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

    evaluar_svm_final(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()