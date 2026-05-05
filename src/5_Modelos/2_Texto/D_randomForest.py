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
# Configuración común
# ──────────────────────────────────────────────────────────────────────────────
N_FOLDS      = 5    # folds para la cross-validation dentro de cada combinación
RANDOM_STATE = 42

# Optuna
N_TRIALS     = 30   # número de combinaciones que explorará Optuna

# Grid search
GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [None, 20, 40],
    "min_samples_leaf": [1, 3, 5],
    "max_features":     ["sqrt", "log2"],
    "class_weight":     ["balanced", None],
}


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


def _cv_score(model: RandomForestClassifier, X: np.ndarray, y: pd.Series) -> tuple[float, float]:
    """Evalúa un modelo con StratifiedKFold y devuelve (media, std) de F1-macro."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


# ──────────────────────────────────────────────────────────────────────────────
# Estrategia 1 — Grid Search
# ──────────────────────────────────────────────────────────────────────────────
def buscar_con_grid(emb_train_val: np.ndarray, y_train_val: pd.Series, run) -> dict:
    """Realiza una búsqueda exhaustiva en GRID con CV estratificada.

    Args:
        emb_train_val: Embeddings de train+val concatenados, shape (n, 384).
        y_train_val:   Etiquetas de train+val.
        run:           Objeto wandb.Run activo para loguear cada combinación.

    Returns:
        dict: Mejores hiperparámetros encontrados.
    """
    from itertools import product

    keys = list(GRID.keys())
    combinaciones = list(product(*[GRID[k] for k in keys]))
    print(f"Iniciando Grid Search ({len(combinaciones)} combinaciones, {N_FOLDS}-fold CV)...")

    mejor_params = None
    mejor_f1 = -1.0

    for i, valores in enumerate(combinaciones):
        params = dict(zip(keys, valores))

        model = RandomForestClassifier(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        f1_mean, f1_std = _cv_score(model, emb_train_val, y_train_val)

        run.log({
            "trial":      i,
            "f1_cv_mean": f1_mean,
            "f1_cv_std":  f1_std,
            **{f"param/{k}": str(v) for k, v in params.items()},
        })
        print(f"  [{i+1}/{len(combinaciones)}] F1={f1_mean:.4f} ± {f1_std:.4f}  {params}")

        if f1_mean > mejor_f1:
            mejor_f1 = f1_mean
            mejor_params = params

    print(f"\n✅ Mejor combinación (F1-CV={mejor_f1:.4f}):\n{mejor_params}")
    return mejor_params, mejor_f1


# ──────────────────────────────────────────────────────────────────────────────
# Estrategia 2 — Optuna
# ──────────────────────────────────────────────────────────────────────────────
def buscar_con_optuna(emb_train_val: np.ndarray, y_train_val: pd.Series, run) -> dict:
    """Búsqueda bayesiana con Optuna (TPE Sampler) sobre rangos continuos/enteros.

    Args:
        emb_train_val: Embeddings de train+val concatenados, shape (n, 384).
        y_train_val:   Etiquetas de train+val.
        run:           Objeto wandb.Run activo para loguear cada trial.

    Returns:
        dict: Mejores hiperparámetros encontrados.
    """
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
        f1_mean, f1_std = _cv_score(model, emb_train_val, y_train_val)

        run.log({
            "trial":      trial.number,
            "f1_cv_mean": f1_mean,
            "f1_cv_std":  f1_std,
            **{f"param/{k}": str(v) for k, v in params.items()},
        })
        return f1_mean

    print(f"Iniciando búsqueda Optuna ({N_TRIALS} trials, {N_FOLDS}-fold CV)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objetivo, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n✅ Mejores hiperparámetros (F1-CV={study.best_value:.4f}):\n{study.best_params}")
    return study.best_params, study.best_value


# ──────────────────────────────────────────────────────────────────────────────
# Función principal de entrenamiento
# ──────────────────────────────────────────────────────────────────────────────
def entrenar_rf_texto(
    emb_train: np.ndarray,
    y_train: pd.Series,
    emb_val: np.ndarray,
    y_val: pd.Series,
    emb_test: np.ndarray,
    y_test: pd.Series,
    estrategia: str = "optuna",
) -> tuple:
    """Optimiza un RandomForest con la estrategia elegida y evalúa en test.

    Flujo:
    1. Concatena train+val → base de optimización con CV estratificada.
    2. Búsqueda de hiperparámetros (grid u optuna) → mejor combinación.
    3. El mejor modelo se reentrena sobre train+val completo.
    4. Evaluación final + classification report + matriz de confusión.
    5. Resultados registrados en Weights & Biases.

    Args:
        emb_train, y_train: Datos de entrenamiento.
        emb_val,   y_val:   Datos de validación.
        emb_test,  y_test:  Datos de test.
        estrategia: "grid" para grid search exhaustivo o "optuna" para TPE.

    Returns:
        tuple: (mejor_modelo, mejor_resultado)
    """
    if estrategia not in {"grid", "optuna"}:
        raise ValueError(f"Estrategia desconocida: {estrategia}. Usa 'grid' u 'optuna'.")

    # Train+val juntos para la CV — el test permanece intacto
    emb_train_val = np.vstack([emb_train, emb_val])
    y_train_val   = pd.concat([y_train, y_val], ignore_index=True)

    run = wandb.init(
        entity="pd1-c2526-team2",
        project="modelo-texto",
        name=f"rf-embeddings-{estrategia}",
        job_type="hyperparameter-search-and-eval",
        group="texto_modelos_embeddings",
        config={
            "modelo":          "RandomForestClassifier",
            "vectorizer":      "sentence-transformers",
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "search_strategy": estrategia,
            "n_trials":        N_TRIALS if estrategia == "optuna" else None,
            "grid":            GRID if estrategia == "grid" else None,
            "cv_folds":        N_FOLDS,
            "random_state":    RANDOM_STATE,
        },
    )

    # ── Búsqueda de hiperparámetros ───────────────────────────────────────────
    if estrategia == "grid":
        best_params, best_f1_cv = buscar_con_grid(emb_train_val, y_train_val, run)
    else:
        best_params, best_f1_cv = buscar_con_optuna(emb_train_val, y_train_val, run)

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
        "best_f1_cv":           best_f1_cv,
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
    """Función principal: carga datos, pre-computa embeddings y lanza el entrenamiento.

    Pregunta al usuario qué estrategia de búsqueda usar antes de empezar.
    """
    nltk.download("stopwords")

    df = bajar_df_texto()
    X, y = x_y_split(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    wandb.login()

    # Pre-computar embeddings UNA sola vez (reutiliza caché si existe)
    emb_train, emb_val, emb_test = obtener_o_cargar_embeddings(X_train, X_val, X_test)

    modo = input("Selecciona modo: 1 (Grid Search) / 2 (Optuna): ").strip()
    if modo == "1":
        estrategia = "grid"
    elif modo == "2":
        estrategia = "optuna"
    else:
        print("Opción no válida. Usando Optuna por defecto.")
        estrategia = "optuna"

    entrenar_rf_texto(
        emb_train, y_train, emb_val, y_val, emb_test, y_test,
        estrategia=estrategia,
    )


if __name__ == "__main__":
    main()