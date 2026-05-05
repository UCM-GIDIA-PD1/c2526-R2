"""
B_nb_embeddings.py - Gaussian Naive Bayes sobre embeddings de Sentence Transformers.

Sigue el esquema unificado del proyecto "modelo-texto-final".
Schema documentado en C_logisticRegression.py.
"""

import numpy as np
import pandas as pd
import wandb

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from funciones_texto import (
    bajar_df_texto, x_y_split, train_val_test_split, obtener_o_cargar_embeddings,
    loguear_resultados_test,
)


# =====================================================================
# CONFIGURACIÓN GLOBAL DE WANDB
# =====================================================================
WANDB_ENTITY = "pd1-c2526-team2"
WANDB_PROJECT = "modelo-texto-final"
WANDB_GROUP = "texto"
MODEL_TYPE = "naive_bayes_embeddings"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
RANDOM_STATE = 42


# =====================================================================
# ENTRENAMIENTO: GRID SEARCH CON HOLDOUT (PredefinedSplit)
# =====================================================================

def optimizar_evaluar_naive_bayes_embeddings(
    emb_train: np.ndarray,
    y_train: pd.Series,
    emb_val: np.ndarray,
    y_val: pd.Series,
    emb_test: np.ndarray,
    y_test: pd.Series,
    x_test_texts: pd.Series,
) -> GaussianNB:
    """Busca hiperparámetros para Gaussian NB sobre embeddings y evalúa en test.

    La búsqueda respeta el split original (PredefinedSplit) para evitar leakage.

    Args:
        emb_train, emb_val, emb_test: embeddings (np.ndarray) de cada split.
        y_train, y_val, y_test:       etiquetas correspondientes.
        x_test_texts:                 textos originales de test, necesarios para el
                                      análisis por longitud (el modelo predice sobre
                                      embeddings, no sobre texto).
    """
    print("Configurando Gaussian Naive Bayes sobre embeddings...")

    # PredefinedSplit: -1 = train (no se valida), 0 = val (se valida)
    emb_train_val = np.vstack([emb_train, emb_val])
    y_train_val = pd.concat([y_train, y_val])
    test_fold = np.concatenate([
        np.full(len(emb_train), -1),
        np.full(len(emb_val), 0),
    ])
    ps = PredefinedSplit(test_fold)

    # var_smoothing controla la estabilidad numérica (evita varianza cero)
    param_grid = {
        "var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3],
    }

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=f"{MODEL_TYPE}-grid_search",
        job_type="hyperparameter-search-and-eval",
        tags=[MODEL_TYPE, "naive_bayes", "embeddings", "grid_search_holdout"],
        config={
            "model_type": MODEL_TYPE,
            "search_strategy": "grid_search_holdout",
            "cv_n_splits": 1,
            "random_state": RANDOM_STATE,
            "vectorizer": "sentence_transformer",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": int(emb_train.shape[1]),
            "search_space": {
                "var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3],
            },
        },
        reinit=True,
    )

    print("\nIniciando GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=GaussianNB(),
        param_grid=param_grid,
        cv=ps,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(emb_train_val, y_train_val)

    best_model = grid_search.best_estimator_
    print(f"\n✅ Mejores hiperparámetros: {grid_search.best_params_}")

    # Tabla con todos los trials del grid search
    cv_table = wandb.Table(columns=[
        "trial_id", "var_smoothing", "cv_f1_macro_mean", "cv_f1_macro_std",
    ])
    cvres = grid_search.cv_results_
    for i in range(len(cvres["params"])):
        cv_table.add_data(
            i,
            cvres["params"][i]["var_smoothing"],
            float(cvres["mean_test_score"][i]),
            float(cvres["std_test_score"][i]),
        )
    wandb.log({"search/cv_results_table": cv_table})

    # Mejor score en CV (con holdout no hay desviación)
    wandb.log({
        "cv/f1_macro_mean": float(grid_search.best_score_),
        "cv/f1_macro_std": 0.0,
    })
    wandb.config.update({
        "best_params": {
            "var_smoothing": grid_search.best_params_["var_smoothing"],
        }
    })

    # Evaluación final en test (esquema unificado).
    # Pasamos x_test_texts aparte porque el modelo come embeddings,
    # no texto, pero el análisis por longitud necesita los textos.
    loguear_resultados_test(best_model, emb_test, y_test, X_test_texts=x_test_texts)

    run.finish()
    return best_model


if __name__ == "__main__":
    df = bajar_df_texto()
    x, y = x_y_split(df)
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y)

    # Pre-computar embeddings UNA sola vez para los tres splits (con cache)
    print("Pre-computando embeddings (cache en disco)...")
    emb_train, emb_val, emb_test = obtener_o_cargar_embeddings(x_train, x_val, x_test)

    optimizar_evaluar_naive_bayes_embeddings(
        emb_train, y_train,
        emb_val, y_val,
        emb_test, y_test,
        x_test_texts=x_test,
    )