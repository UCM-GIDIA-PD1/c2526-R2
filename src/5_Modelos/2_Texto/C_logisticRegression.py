import wandb
import numpy as np
import pandas as pd
import optuna

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone

import nltk
from nltk.corpus import stopwords

from funciones_texto import bajar_df_texto, x_y_split, evaluar_modelo, analizar_por_longitud, loguear_resultados_test


WANDB_ENTITY = "pd1-c2526-team2"
WANDB_PROJECT = "modelo-texto-final"
WANDB_GROUP = "texto"
MODEL_TYPE = "logistic_regression"
RANDOM_STATE = 42
CV_N_SPLITS = 5


# =====================================================================
# ENTRENAMIENTO: GRID SEARCH MANUAL CON STRATIFIED K-FOLD
# =====================================================================

def entrenar_logreg_texto(X_train, y_train, X_test, y_test):
    """Búsqueda en grilla con StratifiedKFold + evaluación final en test."""
    spanish_stopwords = stopwords.words("spanish")

    vectorizers = {"count": CountVectorizer, "tfidf": TfidfVectorizer}
    Cs = [0.1, 1.0, 5.0]
    ngrams = [(1, 1), (1, 2)]
    max_features_list = [5000, 10000]

    skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=f"{MODEL_TYPE}-grid_search",
        job_type="hyperparameter-search-and-eval",
        tags=[MODEL_TYPE, "grid_search_kfold"],
        config={
            "model_type": MODEL_TYPE,
            "search_strategy": "grid_search_kfold",
            "cv_n_splits": CV_N_SPLITS,
            "random_state": RANDOM_STATE,
            "search_space": {
                "vectorizers": list(vectorizers.keys()),
                "C": Cs,
                "ngram_range": [str(n) for n in ngrams],
                "max_features": max_features_list,
            },
        },
        reinit=True,
    )

    cv_table = wandb.Table(columns=[
        "trial_id", "vectorizer", "C", "ngram", "max_features",
        "cv_f1_macro_mean", "cv_f1_macro_std",
    ])

    print("Buscando mejor modelo de texto (grid search + CV)...")
    mejor_resultado = None
    mejor_config = None
    trial_id = 0

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
                            stop_words=spanish_stopwords,
                        )),
                        ("classifier", LogisticRegression(
                            C=C,
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        )),
                    ])

                    cv_scores = []
                    for train_idx, val_idx in skf.split(X_train, y_train):
                        X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
                        m = clone(model)
                        m.fit(X_tr, y_tr)
                        cv_scores.append(
                            f1_score(y_vl, m.predict(X_vl), average="macro")
                        )

                    f1_mean = float(np.mean(cv_scores))
                    f1_std = float(np.std(cv_scores))
                    print(f"  cv_f1_macro = {f1_mean:.4f} ± {f1_std:.4f}")

                    cv_table.add_data(
                        trial_id, vec_name, C, str(ngram), max_feat, f1_mean, f1_std,
                    )

                    if (mejor_resultado is None) or (f1_mean > mejor_resultado["cv_f1_macro_mean"]):
                        mejor_resultado = {
                            "trial_id": trial_id,
                            "nombre": nombre,
                            "vectorizer": vec_name,
                            "C": C,
                            "ngram": list(ngram),
                            "max_features": max_feat,
                            "cv_f1_macro_mean": f1_mean,
                            "cv_f1_macro_std": f1_std,
                        }
                        mejor_config = {
                            "vec_class": vec_class,
                            "vectorizer": vec_name,
                            "C": C,
                            "ngram": ngram,
                            "max_features": max_feat,
                        }
                    trial_id += 1

    print("\n=== MEJOR MODELO (CV) ===")
    print(mejor_resultado)

    # Reentrenar con todo el train usando la mejor config
    mejor_modelo = Pipeline([
        ("vectorizer", mejor_config["vec_class"](
            max_features=mejor_config["max_features"],
            ngram_range=mejor_config["ngram"],
            stop_words=spanish_stopwords,
        )),
        ("classifier", LogisticRegression(
            C=mejor_config["C"],
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    mejor_modelo.fit(X_train, y_train)

    # Logueamos los resultados de la búsqueda
    wandb.log({"search/cv_results_table": cv_table})
    wandb.log({
        "cv/f1_macro_mean": mejor_resultado["cv_f1_macro_mean"],
        "cv/f1_macro_std": mejor_resultado["cv_f1_macro_std"],
    })
    wandb.config.update({
        "best_params": {
            "vectorizer": mejor_config["vectorizer"],
            "C": mejor_config["C"],
            "ngram_range": str(mejor_config["ngram"]),
            "max_features": mejor_config["max_features"],
        }
    })

    # Evaluación final en test (esquema unificado)
    loguear_resultados_test(mejor_modelo, X_test, y_test)

    run.finish()
    return mejor_modelo, mejor_resultado


# =====================================================================
# ENTRENAMIENTO: BÚSQUEDA CON OPTUNA
# =====================================================================

def entrenar_logreg_texto_optuna(X_train, y_train, X_test, y_test, n_trials: int = 20):
    """Búsqueda con Optuna + pruning por fold + evaluación final en test."""
    spanish_stopwords = stopwords.words("spanish")
    skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=f"{MODEL_TYPE}-optuna",
        job_type="hyperparameter-search-and-eval",
        tags=[MODEL_TYPE, "optuna"],
        config={
            "model_type": MODEL_TYPE,
            "search_strategy": "optuna",
            "cv_n_splits": CV_N_SPLITS,
            "random_state": RANDOM_STATE,
            "n_trials": n_trials,
            "search_space": {
                "vectorizer": ["count", "tfidf"],
                "C": [0.1, 5.0],
                "ngram": ["(1,1)", "(1,2)"],
                "max_features": [1000, 15000],
            },
        },
        reinit=True,
    )

    def objective(trial):
        vec_name = trial.suggest_categorical("vectorizer", ["count", "tfidf"])
        C = trial.suggest_float("C", 0.1, 5.0)
        ngram = trial.suggest_categorical("ngram", [(1, 1), (1, 2)])
        max_features = trial.suggest_int("max_features", 1000, 15000, step=1000)

        vec_class = CountVectorizer if vec_name == "count" else TfidfVectorizer
        model = Pipeline([
            ("vectorizer", vec_class(
                max_features=max_features,
                ngram_range=ngram,
                stop_words=spanish_stopwords,
            )),
            ("classifier", LogisticRegression(
                C=C, max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])

        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
            m = clone(model)
            m.fit(X_tr, y_tr)
            scores.append(f1_score(y_vl, m.predict(X_vl), average="macro"))
            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    print("Buscando mejor modelo con Optuna...")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials)

    # Loguear historial de trials
    optuna_table = wandb.Table(columns=[
        "trial_id", "vectorizer", "C", "ngram", "max_features",
        "cv_f1_macro_mean", "state",
    ])
    for t in study.trials:
        params = t.params
        optuna_table.add_data(
            t.number,
            params.get("vectorizer"),
            params.get("C"),
            str(params.get("ngram")),
            params.get("max_features"),
            t.value if t.value is not None else float("nan"),
            str(t.state),
        )
    wandb.log({"search/optuna_trials_table": optuna_table})

    best_params = study.best_params
    print("\n=== MEJOR CONFIG OPTUNA ===")
    print(best_params)

    # Reentrenar con todo el train usando los mejores params
    vec_class = CountVectorizer if best_params["vectorizer"] == "count" else TfidfVectorizer
    mejor_modelo = Pipeline([
        ("vectorizer", vec_class(
            max_features=best_params["max_features"],
            ngram_range=best_params["ngram"],
            stop_words=spanish_stopwords,
        )),
        ("classifier", LogisticRegression(
            C=best_params["C"],
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    mejor_modelo.fit(X_train, y_train)

    wandb.config.update({
        "best_params": {
            "vectorizer": best_params["vectorizer"],
            "C": best_params["C"],
            "ngram_range": str(best_params["ngram"]),
            "max_features": best_params["max_features"],
        }
    })
    wandb.log({"cv/f1_macro_mean": study.best_value})

    # Evaluación final en test (esquema unificado)
    loguear_resultados_test(mejor_modelo, X_test, y_test)

    run.finish()
    return mejor_modelo, best_params


# =====================================================================
# MAIN
# =====================================================================

def main():
    """Carga datos, divide en train/test y lanza el entrenamiento elegido."""
    nltk.download("stopwords", quiet=True)

    df = bajar_df_texto()
    X, y = x_y_split(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    wandb.login()

    modo = input("Selecciona modo: 1 (Grid Search) / 2 (Optuna): ")
    if modo == "1":
        entrenar_logreg_texto(X_train, y_train, X_test, y_test)
    elif modo == "2":
        entrenar_logreg_texto_optuna(X_train, y_train, X_test, y_test)
    else:
        print("Opción no válida")


if __name__ == "__main__":
    main()