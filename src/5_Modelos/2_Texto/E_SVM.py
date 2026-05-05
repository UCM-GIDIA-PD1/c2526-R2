import wandb
import numpy as np
import pandas as pd
import optuna
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone

import nltk
from nltk.corpus import stopwords

from funciones_texto import bajar_df_texto, x_y_split, loguear_resultados_test


WANDB_ENTITY = "pd1-c2526-team2"
WANDB_PROJECT = "modelo-texto-final"
WANDB_GROUP = "texto"
MODEL_TYPE = "svm"
RANDOM_STATE = 42
CV_N_SPLITS = 5

ARTIFACT_NAME = "mejor-svm-texto"
MODEL_FILENAME = "mejor_SVM_texto.joblib"

# Optuna no admite tuplas en suggest_categorical (las quiere como str/int/float/bool/None).
# Almacenamos los ngrams como string y los mapeamos a tupla al usarlos.
NGRAM_CHOICES = {"(1,1)": (1, 1), "(1,2)": (1, 2)}


# =====================================================================
# ENTRENAMIENTO: GRID SEARCH MANUAL CON STRATIFIED K-FOLD
# =====================================================================

def entrenar_svm_texto(X_train, y_train, X_test, y_test):
    """Búsqueda en grilla con StratifiedKFold + evaluación final en test."""
    spanish_stopwords = stopwords.words("spanish")

    Cs = [0.1, 1.0, 10.0]
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
                "vectorizer": "tfidf",
                "C": Cs,
                "ngram_range": [str(n) for n in ngrams],
                "max_features": max_features_list,
            },
        },
        reinit=True,
    )

    cv_table = wandb.Table(columns=[
        "trial_id", "C", "ngram", "max_features",
        "cv_f1_macro_mean", "cv_f1_macro_std",
    ])

    print("Buscando mejor modelo SVM (grid search + CV)...")
    mejor_resultado = None
    mejor_config = None
    trial_id = 0

    for C in Cs:
        for ngram in ngrams:
            for max_feat in max_features_list:
                nombre = f"svm_C{C}_ng{ngram}_mf{max_feat}"
                print(f"\nEntrenando: {nombre}")

                model = Pipeline([
                    ("vectorizer", TfidfVectorizer(
                        max_features=max_feat,
                        ngram_range=ngram,
                        stop_words=spanish_stopwords,
                    )),
                    ("classifier", LinearSVC(
                        C=C,
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
                    trial_id, C, str(ngram), max_feat, f1_mean, f1_std,
                )

                if (mejor_resultado is None) or (f1_mean > mejor_resultado["cv_f1_macro_mean"]):
                    mejor_resultado = {
                        "trial_id": trial_id,
                        "nombre": nombre,
                        "vectorizer": "tfidf",
                        "C": C,
                        "ngram": list(ngram),
                        "max_features": max_feat,
                        "cv_f1_macro_mean": f1_mean,
                        "cv_f1_macro_std": f1_std,
                    }
                    mejor_config = {
                        "C": C,
                        "ngram": ngram,
                        "max_features": max_feat,
                    }
                trial_id += 1

    print("\n=== MEJOR MODELO (CV) ===")
    print(mejor_resultado)

    # Reentrenar con todo el train usando la mejor config
    mejor_modelo = Pipeline([
        ("vectorizer", TfidfVectorizer(
            max_features=mejor_config["max_features"],
            ngram_range=mejor_config["ngram"],
            stop_words=spanish_stopwords,
        )),
        ("classifier", LinearSVC(
            C=mejor_config["C"],
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
            "vectorizer": "tfidf",
            "C": mejor_config["C"],
            "ngram_range": str(mejor_config["ngram"]),
            "max_features": mejor_config["max_features"],
        }
    })

    # Evaluación final en test (esquema unificado)
    loguear_resultados_test(mejor_modelo, X_test, y_test)

    # Guardar el modelo como artifact en W&B
    _guardar_artifact(mejor_modelo, mejor_resultado, run)

    run.finish()
    return mejor_modelo, mejor_resultado


# =====================================================================
# ENTRENAMIENTO: BÚSQUEDA CON OPTUNA
# =====================================================================

def entrenar_svm_texto_optuna(X_train, y_train, X_test, y_test, n_trials: int = 30):
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
                "vectorizer": ["tfidf"],
                "C": [0.01, 10.0],
                "ngram": ["(1,1)", "(1,2)"],
                "max_features": [1000, 20000],
                "max_df": [0.7, 1.0],
                "min_df": [1, 10],
            },
        },
        reinit=True,
    )

    def objective(trial):
        C = trial.suggest_float("C", 0.01, 10.0, log=True)
        ngram_str = trial.suggest_categorical("ngram", list(NGRAM_CHOICES.keys()))
        ngram = NGRAM_CHOICES[ngram_str]
        max_features = trial.suggest_int("max_features", 1000, 20000, step=1000)
        max_df = trial.suggest_float("max_df", 0.7, 1.0)
        min_df = trial.suggest_int("min_df", 1, 10)

        model = Pipeline([
            ("vectorizer", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram,
                max_df=max_df,
                min_df=min_df,
                stop_words=spanish_stopwords,
            )),
            ("classifier", LinearSVC(
                C=C,
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

    print("Buscando mejor modelo SVM con Optuna...")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials)

    # Loguear historial de trials
    optuna_table = wandb.Table(columns=[
        "trial_id", "vectorizer", "C", "ngram", "max_features",
        "max_df", "min_df", "cv_f1_macro_mean", "state",
    ])
    for t in study.trials:
        params = t.params
        optuna_table.add_data(
            t.number,
            "tfidf",
            params.get("C"),
            str(params.get("ngram")),
            params.get("max_features"),
            params.get("max_df"),
            params.get("min_df"),
            t.value if t.value is not None else float("nan"),
            str(t.state),
        )
    wandb.log({"search/optuna_trials_table": optuna_table})

    best_params = study.best_params
    print("\n=== MEJOR CONFIG OPTUNA ===")
    print(best_params)

    # best_params["ngram"] es un string como "(1,1)"; lo recuperamos como tupla
    best_ngram = NGRAM_CHOICES[best_params["ngram"]]

    # Reentrenar con todo el train usando los mejores params
    mejor_modelo = Pipeline([
        ("vectorizer", TfidfVectorizer(
            max_features=best_params["max_features"],
            ngram_range=best_ngram,
            max_df=best_params["max_df"],
            min_df=best_params["min_df"],
            stop_words=spanish_stopwords,
        )),
        ("classifier", LinearSVC(
            C=best_params["C"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    mejor_modelo.fit(X_train, y_train)

    wandb.config.update({
        "best_params": {
            "vectorizer": "tfidf",
            "C": best_params["C"],
            "ngram_range": best_params["ngram"],
            "max_features": best_params["max_features"],
            "max_df": best_params["max_df"],
            "min_df": best_params["min_df"],
        }
    })
    wandb.log({"cv/f1_macro_mean": study.best_value})

    # Evaluación final en test (esquema unificado)
    loguear_resultados_test(mejor_modelo, X_test, y_test)

    # Guardar el modelo como artifact en W&B
    mejor_resultado = {
        "vectorizer": "tfidf",
        "C": best_params["C"],
        "ngram": list(best_ngram),
        "max_features": best_params["max_features"],
        "max_df": best_params["max_df"],
        "min_df": best_params["min_df"],
        "cv_f1_macro_mean": study.best_value,
    }
    _guardar_artifact(mejor_modelo, mejor_resultado, run)

    run.finish()
    return mejor_modelo, best_params


# =====================================================================
# ARTIFACT
# =====================================================================

def _guardar_artifact(modelo, mejor_resultado, run):
    """Serializa el pipeline con joblib y lo sube como artifact al run actual."""
    print(f"\nGuardando modelo como artifact: {MODEL_FILENAME}")
    joblib.dump(modelo, MODEL_FILENAME)

    artifact = wandb.Artifact(
        name=ARTIFACT_NAME,
        type="model",
        description="Mejor modelo SVM (LinearSVC + TfidfVectorizer)",
        metadata=mejor_resultado,
    )
    artifact.add_file(MODEL_FILENAME)
    run.log_artifact(artifact)
    print("Artifact subido correctamente.")


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
        entrenar_svm_texto(X_train, y_train, X_test, y_test)
    elif modo == "2":
        entrenar_svm_texto_optuna(X_train, y_train, X_test, y_test)
    else:
        print("Opción no válida")


if __name__ == "__main__":
    main()