import wandb
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, confusion_matrix
)
import nltk

from funciones_texto import (
    bajar_df_texto, x_y_split, train_val_test_split, obtener_o_cargar_embeddings
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────────────────
N_TRIALS     = 30   # número de combinaciones que explorará Optuna
N_FOLDS      = 5    # folds para la cross-validation dentro de cada trial
RANDOM_STATE = 42


def evaluar_modelo(model, X: np.ndarray, y: pd.Series) -> dict:
    """Evalúa un modelo ya entrenado sobre un conjunto dado.

    Args:
        model: Modelo entrenado con método ``predict``.
        X:     Embeddings, shape (n, 384).
        y:     Etiquetas reales.

    Returns:
        dict: accuracy, f1_macro, recall_macro, precision_macro.
    """
    y_pred = model.predict(X)
    return {
        "accuracy":        accuracy_score(y, y_pred),
        "f1_macro":        f1_score(y, y_pred, average="macro"),
        "recall_macro":    recall_score(y, y_pred, average="macro"),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
    }


def construir_objetivo(emb_train_val: np.ndarray, y_train_val: pd.Series, run) -> callable:
    """Devuelve la función objetivo que Optuna maximizará.

    Cada llamada entrena un RandomForest con los hiperparámetros sugeridos y
    devuelve la media del F1-macro en CV estratificada. Loguea cada trial en
    Weights & Biases para seguimiento en tiempo real.

    No se incluye StandardScaler: los árboles de decisión son invariantes a la
    escala de las features, por lo que no aportaría nada.

    Args:
        emb_train_val: Embeddings de train+val concatenados, shape (n, 384).
        y_train_val:   Etiquetas de train+val.
        run:           Objeto wandb.Run activo.

    Returns:
        callable: Función objetivo compatible con optuna.Study.optimize().
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objetivo(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
            "max_depth":        trial.suggest_int("max_depth", 5, 60, step=5),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            "class_weight":     trial.suggest_categorical("class_weight", ["balanced", None]),
        }

        model = RandomForestClassifier(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        # F1-macro medio sobre los N_FOLDS folds
        scores = cross_val_score(
            model, emb_train_val, y_train_val,
            cv=cv, scoring="f1_macro", n_jobs=-1,
        )
        f1_cv_mean = float(scores.mean())
        f1_cv_std  = float(scores.std())

        # Log por trial en W&B
        run.log({
            "trial":      trial.number,
            "f1_cv_mean": f1_cv_mean,
            "f1_cv_std":  f1_cv_std,
            **{f"param/{k}": str(v) for k, v in params.items()},
        })

        return f1_cv_mean

    return objetivo


def entrenar_rf_texto(
    emb_train: np.ndarray,
    y_train: pd.Series,
    emb_val: np.ndarray,
    y_val: pd.Series,
    emb_test: np.ndarray,
    y_test: pd.Series,
) -> tuple:
    """Optimiza un RandomForest con Optuna + CV sobre embeddings y evalúa el
    mejor modelo en test.

    Flujo:
    1. Concatena train+val → base de optimización con CV estratificada.
    2. Optuna busca los mejores hiperparámetros en N_TRIALS trials.
    3. El mejor modelo se reentrena sobre train+val completo.
    4. Evaluación final + classification report + matriz de confusión.
    5. Resultados registrados en Weights & Biases.

    Args:
        emb_train: Embeddings de entrenamiento, shape (n_train, 384).
        y_train:   Etiquetas de entrenamiento.
        emb_val:   Embeddings de validación, shape (n_val, 384).
        y_val:     Etiquetas de validación.
        emb_test:  Embeddings de test, shape (n_test, 384).
        y_test:    Etiquetas de test.

    Returns:
        tuple: (mejor_modelo, mejor_resultado)
    """
    # Train+val juntos para la CV — el test permanece intacto
    emb_train_val = np.vstack([emb_train, emb_val])
    y_train_val   = pd.concat([y_train, y_val], ignore_index=True)

    run = wandb.init(
        entity="pd1-c2526-team2",
        project="modelo-texto",
        name="rf-embeddings-optuna",
        job_type="hyperparameter-search-and-eval",
        group="texto_modelos_embeddings",
        config={
            "modelo":          "RandomForestClassifier",
            "vectorizer":      "sentence-transformers",
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "search_strategy": "Optuna",
            "n_trials":        N_TRIALS,
            "cv_folds":        N_FOLDS,
            "random_state":    RANDOM_STATE,
        },
    )

    # ── Búsqueda con Optuna ───────────────────────────────────────────────────
    print(f"Iniciando búsqueda Optuna ({N_TRIALS} trials, {N_FOLDS}-fold CV)...")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        construir_objetivo(emb_train_val, y_train_val, run),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_f1_cv  = study.best_value
    print(f"\n✅ Mejores hiperparámetros (F1-CV={best_f1_cv:.4f}):\n{best_params}")

    # ── Reentrenar sobre train+val completo con los mejores params ────────────
    mejor_modelo = RandomForestClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    mejor_modelo.fit(emb_train_val, y_train_val)

    # ── Evaluación final en test ──────────────────────────────────────────────
    metricas_test = evaluar_modelo(mejor_modelo, emb_test, y_test)
    y_pred_test = mejor_modelo.predict(emb_test)

    print("\n=== RESULTADOS EN TEST ===")
    print(metricas_test)

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred_test))

    print("\n=== MATRIZ DE CONFUSIÓN ===")
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred_test, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    wandb.log({
        "best_f1_cv":     best_f1_cv,
        "test_f1_macro":        metricas_test["f1_macro"],
        "test_accuracy":        metricas_test["accuracy"],
        "test_recall_macro":    metricas_test["recall_macro"],
        "test_precision_macro": metricas_test["precision_macro"],
        **{f"best_param/{k}": str(v) for k, v in best_params.items()},
    })

    run.finish()

    mejor_resultado = metricas_test | {"best_params": best_params, "best_f1_cv": best_f1_cv}
    return mejor_modelo, mejor_resultado


def main():
    """Función principal: carga datos, pre-computa embeddings y lanza el entrenamiento."""
    nltk.download("stopwords")

    df = bajar_df_texto()
    X, y = x_y_split(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    wandb.login()

    # Pre-computar embeddings UNA sola vez (reutiliza caché si existe)
    emb_train, emb_val, emb_test = obtener_o_cargar_embeddings(X_train, X_val, X_test)

    entrenar_rf_texto(emb_train, y_train, emb_val, y_val, emb_test, y_test)


if __name__ == "__main__":
    main()