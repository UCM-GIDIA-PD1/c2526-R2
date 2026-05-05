import numpy as np
import pandas as pd
import wandb

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, PredefinedSplit

import nltk
from nltk.corpus import stopwords

from funciones_texto import (
    bajar_df_texto, x_y_split, train_val_test_split,
    loguear_resultados_test,
)


# =====================================================================
# CONFIGURACIÓN GLOBAL DE WANDB
# =====================================================================
WANDB_ENTITY = "pd1-c2526-team2"
WANDB_PROJECT = "modelo-texto-final"
WANDB_GROUP = "texto"
MODEL_TYPE = "naive_bayes_tfidf"
RANDOM_STATE = 42


# =====================================================================
# ENTRENAMIENTO: GRID SEARCH CON HOLDOUT (PredefinedSplit)
# =====================================================================

def optimizar_evaluar_naive_bayes(x_train, y_train, x_val, y_val, x_test, y_test):
    """Busca hiperparámetros para Multinomial NB sobre TF-IDF y evalúa en test.

    Usa PredefinedSplit para hacer grid search respetando el split original
    (train se usa para entrenar y val como fold de validación).
    """
    print("Descargando stopwords y montando pipeline TF-IDF + MultinomialNB...")
    nltk.download("stopwords", quiet=True)
    spanish_stopwords = stopwords.words("spanish")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words=spanish_stopwords)),
        ("clf", MultinomialNB()),
    ])

    param_grid = {
        "tfidf__max_features": [5000, 10000, None],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__alpha": [0.1, 0.5, 1.0, 5.0],
    }

    # PredefinedSplit: -1 = train (no se valida), 0 = val (se valida)
    x_train_val = pd.concat([x_train, x_val])
    y_train_val = pd.concat([y_train, y_val])
    test_fold = np.concatenate([
        np.full(len(x_train), -1),
        np.full(len(x_val), 0),
    ])
    ps = PredefinedSplit(test_fold)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=f"{MODEL_TYPE}-grid_search",
        job_type="hyperparameter-search-and-eval",
        tags=[MODEL_TYPE, "naive_bayes", "tfidf", "grid_search_holdout"],
        config={
            "model_type": MODEL_TYPE,
            "search_strategy": "grid_search_holdout",
            "cv_n_splits": 1,
            "random_state": RANDOM_STATE,
            "vectorizer": "tfidf",
            "search_space": {
                "tfidf__max_features": [5000, 10000, None],
                "tfidf__ngram_range": [str(n) for n in [(1, 1), (1, 2)]],
                "clf__alpha": [0.1, 0.5, 1.0, 5.0],
            },
        },
        reinit=True,
    )

    print("\nIniciando GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=ps,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(x_train_val, y_train_val)

    best_model = grid_search.best_estimator_
    print(f"\n✅ Mejores hiperparámetros: {grid_search.best_params_}")

    # Tabla con todos los trials del grid search
    cv_table = wandb.Table(columns=[
        "trial_id", "params", "cv_f1_macro_mean", "cv_f1_macro_std",
    ])
    cvres = grid_search.cv_results_
    for i in range(len(cvres["params"])):
        cv_table.add_data(
            i,
            str(cvres["params"][i]),
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
            "tfidf__max_features": grid_search.best_params_["tfidf__max_features"],
            "tfidf__ngram_range": str(grid_search.best_params_["tfidf__ngram_range"]),
            "clf__alpha": grid_search.best_params_["clf__alpha"],
        }
    })

    # Evaluación final en test (esquema unificado)
    loguear_resultados_test(best_model, x_test, y_test)

    run.finish()
    return best_model


if __name__ == "__main__":
    df = bajar_df_texto()
    x, y = x_y_split(df)
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y)

    optimizar_evaluar_naive_bayes(x_train, y_train, x_val, y_val, x_test, y_test)