import numpy as np
import pandas as pd
import wandb

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.base import clone

import nltk
from nltk.corpus import stopwords
import optuna

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
# GRID SEARCH + WANDB
# =========================
def entrenar_svm_texto(X_train, y_train, X_test, y_test):

    spanish_stopwords = stopwords.words("spanish")

    run = wandb.init(
        project="modelo-texto-f",
        entity="pd1-c2526-team2",
        name="svm-grid",
        group="svm-grid",
        job_type="grid",
        config={
            "model": "svm",
            "search": "grid"
        }
    )

    vectorizers = {
        "tfidf": TfidfVectorizer
    }

    Cs = [1.0, 10.0]
    ngrams = [(1,1), (1,2)]
    max_features_list = [5000, 10000]
    min_df_list = [3]
    max_df_list = [0.8]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    mejor_resultado = None
    mejor_modelo = None

    print("Buscando mejor SVM (Grid)...")

    for vec_name, vec_class in vectorizers.items():
        for C in Cs:
            for ngram in ngrams:
                for max_feat in max_features_list:
                    for min_df in min_df_list:
                        for max_df in max_df_list:

                            nombre = f"{vec_name}_C{C}_ng{ngram}_mf{max_feat}_min{min_df}_max{max_df}"

                            model = Pipeline([
                                ("vectorizer", vec_class(
                                    max_features=max_feat,
                                    ngram_range=ngram,
                                    min_df=min_df,
                                    max_df=max_df,
                                    stop_words=spanish_stopwords
                                )),
                                ("classifier", LinearSVC(
                                    C=C,
                                    class_weight="balanced",
                                    random_state=42
                                ))
                            ])

                            cv_scores = []

                            for train_idx, val_idx in skf.split(X_train, y_train):

                                X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
                                y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]

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
                                "vectorizer": vec_name,
                                "C": C,
                                "ngram": str(ngram),
                                "max_features": max_feat,
                                "min_df": min_df,
                                "max_df": max_df
                            })

                            if (mejor_resultado is None) or (f1_mean > mejor_resultado["f1_macro"]):
                                mejor_resultado = {
                                    "nombre": nombre,
                                    "vectorizer": vec_name,
                                    "C": C,
                                    "ngram": ngram,
                                    "max_features": max_feat,
                                    "min_df": min_df,
                                    "max_df": max_df,
                                    "f1_macro": f1_mean,
                                    "f1_std": f1_std
                                }

                                mejor_config = {
                                    "vec_class": vec_class,
                                    "C": C,
                                    "ngram": ngram,
                                    "max_features": max_feat,
                                    "min_df": min_df,
                                    "max_df": max_df
                                }

    print("\n=== MEJOR MODELO SVM ===")
    print(mejor_resultado)

    wandb.log({
        "best_cv_f1": mejor_resultado["f1_macro"],
        "best_model": mejor_resultado["nombre"]
    })

    mejor_modelo = Pipeline([
        ("vectorizer", mejor_config["vec_class"](
            max_features=mejor_config["max_features"],
            ngram_range=mejor_config["ngram"],
            min_df=mejor_config["min_df"],
            max_df=mejor_config["max_df"],
            stop_words=spanish_stopwords
        )),
        ("classifier", LinearSVC(
            C=mejor_config["C"],
            class_weight="balanced",
            random_state=42
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

    print("\n=== ANALISIS POR CLASE ===")
    print(classification_report(y_test, mejor_modelo.predict(X_test)))

    print("\n=== ANALISIS POR LONGITUD ===")
    print(analizar_por_longitud(mejor_modelo, X_test, y_test))

    cm = confusion_matrix(y_test, mejor_modelo.predict(X_test))

    wandb.log({
        "confusion_matrix": wandb.Table(
            data=cm.tolist(),
            columns=[str(c) for c in np.unique(y_test)]
        )
    })

    wandb.run.summary["best_cv_f1"] = mejor_resultado["f1_macro"]
    wandb.run.summary["best_model_name"] = mejor_resultado["nombre"]

    wandb.run.summary["best_params"] = {
        "vectorizer": mejor_resultado["vectorizer"],
        "C": mejor_resultado["C"],
        "ngram": str(mejor_resultado["ngram"]),
        "max_features": mejor_resultado["max_features"],
        "min_df": mejor_resultado["min_df"],
        "max_df": mejor_resultado["max_df"]
    }
    run.finish()

    return mejor_modelo, mejor_resultado


# =========================
# OPTUNA + WANDB
# =========================
def entrenar_svm_texto_optuna(X_train, y_train, X_test, y_test):

    spanish_stopwords = stopwords.words("spanish")

    run = wandb.init(
        project="modelo-texto-f",
        entity="pd1-c2526-team2",
        name="svm-optuna",
        group="svm-optuna",
        job_type="optuna",
        config={
            "model": "svm",
            "search": "optuna"
        }
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):

        vec_name = trial.suggest_categorical("vectorizer", ["count", "tfidf"])
        vec_class = CountVectorizer if vec_name == "count" else TfidfVectorizer

        C = trial.suggest_float("C", 0.01, 10.0, log=True)
        ngram = trial.suggest_categorical("ngram", [(1,1), (1,2)])
        max_features = trial.suggest_int("max_features", 2000, 20000, step=2000)
        min_df = trial.suggest_int("min_df", 1, 5)
        max_df = trial.suggest_float("max_df", 0.7, 1.0)

        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):

            X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = Pipeline([
                ("vectorizer", vec_class(
                    max_features=max_features,
                    ngram_range=ngram,
                    min_df=min_df,
                    max_df=max_df,
                    stop_words=spanish_stopwords
                )),
                ("classifier", LinearSVC(
                    C=C,
                    class_weight="balanced",
                    random_state=42
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

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )

    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    wandb.log({"best_params": best_params})

    vec_class = CountVectorizer if best_params["vectorizer"] == "count" else TfidfVectorizer

    mejor_modelo = Pipeline([
        ("vectorizer", vec_class(
            max_features=best_params["max_features"],
            ngram_range=best_params["ngram"],
            min_df=best_params["min_df"],
            max_df=best_params["max_df"],
            stop_words=spanish_stopwords
        )),
        ("classifier", LinearSVC(
            C=best_params["C"],
            class_weight="balanced",
            random_state=42
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

    print("\n=== ANALISIS POR CLASE ===")
    print(classification_report(y_test, mejor_modelo.predict(X_test)))

    print("\n=== ANALISIS POR LONGITUD ===")
    print(analizar_por_longitud(mejor_modelo, X_test, y_test))


    wandb.run.summary["best_cv_f1"] = max(study.best_value, 0)

    wandb.run.summary["best_params"] = best_params
    run.finish()

    return mejor_modelo, best_params


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    nltk.download("stopwords")

    df = bajar_df_texto()
    X, y = x_y_split(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    modo = input("Selecciona modo: 1 (Grid) / 2 (Optuna): ")

    if modo == "1":
        entrenar_svm_texto(X_train, y_train, X_test, y_test)
    elif modo == "2":
        entrenar_svm_texto_optuna(X_train, y_train, X_test, y_test)
    else:
        print("Opción no válida")