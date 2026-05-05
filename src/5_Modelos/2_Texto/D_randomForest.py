import numpy as np
import pandas as pd
import optuna
import wandb

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone

import nltk
from nltk.corpus import stopwords

from funciones_texto import bajar_df_texto, x_y_split


def evaluar_modelo(model, X_test, y_test):

    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
    }


def analizar_por_longitud(model, X_test, y_test):

    df = pd.DataFrame({
        "texto": X_test,
        "y_true": y_test
    })

    df["longitud"] = df["texto"].apply(lambda x: len(x.split()))

    bins = [0, 20, 50, 100, np.inf]
    labels = ["corto", "medio", "largo", "muy_largo"]
    df["segmento"] = pd.cut(df["longitud"], bins=bins, labels=labels)

    resultados = []

    for seg in labels:
        subset = df[df["segmento"] == seg]
        if len(subset) == 0:
            continue

        y_pred = model.predict(subset["texto"])
        f1 = f1_score(subset["y_true"], y_pred, average="macro")

        resultados.append({
            "segmento": seg,
            "n_samples": len(subset),
            "f1_macro": f1
        })

    return pd.DataFrame(resultados)


# =========================
# GRID SEARCH + WANDB (1 RUN)
# =========================
def entrenar_rf_texto(X_train, y_train, X_test, y_test):

    spanish_stopwords = stopwords.words("spanish")

    vectorizers = {
        "count": CountVectorizer,
        "tfidf": TfidfVectorizer
    }

    n_estimators_list = [100, 200]
    max_depth_list = [20]
    min_samples_leaf_list = [1]
    max_features_list = [10000]
    ngrams = [(1, 1), (1, 2)]

    mejor_resultado = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    run = wandb.init(
        project="modelo-texto-f",
        group="rf-grid",
        name="rf-grid",
        tags=["rf", "grid"],
        config={"model": "rf", "search": "grid"}
    )

    print("Buscando mejor Random Forest (Grid)...")

    for vec_name, vec_class in vectorizers.items():
        for n_est in n_estimators_list:
            for max_d in max_depth_list:
                for min_leaf in min_samples_leaf_list:
                    for max_feat in max_features_list:
                        for ngram in ngrams:

                            model = Pipeline([
                                ("vectorizer", vec_class(
                                    max_features=max_feat,
                                    ngram_range=ngram,
                                    stop_words=spanish_stopwords
                                )),
                                ("classifier", RandomForestClassifier(
                                    n_estimators=n_est,
                                    max_depth=max_d,
                                    min_samples_leaf=min_leaf,
                                    random_state=42,
                                    n_jobs=-1
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
                                "n_estimators": n_est,
                                "max_depth": max_d,
                                "min_samples_leaf": min_leaf,
                                "max_features": max_feat,
                                "ngram": str(ngram),
                                "vectorizer": vec_name
                            })

                            if (mejor_resultado is None) or (f1_mean > mejor_resultado["f1_macro"]):
                                mejor_resultado = {
                                    "vectorizer": vec_name,
                                    "n_estimators": n_est,
                                    "max_depth": max_d,
                                    "min_samples_leaf": min_leaf,
                                    "max_features": max_feat,
                                    "ngram": ngram,
                                    "f1_macro": f1_mean,
                                    "f1_std": f1_std
                                }
                                mejor_config = {
                                    "vec_class": vec_class,
                                    "n_estimators": n_est,
                                    "max_depth": max_d,
                                    "min_samples_leaf": min_leaf,
                                    "max_features": max_feat,
                                    "ngram": ngram
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
        ("classifier", RandomForestClassifier(
            n_estimators=mejor_config["n_estimators"],
            max_depth=mejor_config["max_depth"],
            min_samples_leaf=mejor_config["min_samples_leaf"],
            random_state=42,
            n_jobs=-1
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
def entrenar_rf_texto_optuna(X_train, y_train, X_test, y_test):

    spanish_stopwords = stopwords.words("spanish")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    run = wandb.init(
        project="modelo-texto-f",
        group="rf-optuna",
        name="rf-optuna",
        tags=["rf", "optuna"],
        config={"model": "rf", "search": "optuna"}
    )

    def objective(trial):

        vec_name = trial.suggest_categorical("vectorizer", ["count", "tfidf"])
        n_estimators = trial.suggest_int("n_estimators", 100, 300, step=100)
        max_depth = trial.suggest_categorical("max_depth", [None, 10, 20, 30])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
        max_features = trial.suggest_int("max_features", 2000, 15000, step=2000)
        ngram = trial.suggest_categorical("ngram", [(1, 1), (1, 2)])

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
                ("classifier", RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1
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
        ("classifier", RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
            random_state=42,
            n_jobs=-1
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
        entrenar_rf_texto(X_train, y_train, X_test, y_test)
    else:
        entrenar_rf_texto_optuna(X_train, y_train, X_test, y_test)