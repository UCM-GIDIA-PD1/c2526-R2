import os
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score
import nltk
from nltk.corpus import stopwords

from utils.funciones_minio import crear_cliente_minio, bajar_minio
from utils.config import PATH_PRIMARIOS_LIMPIO

OBJ_VIVIENDAS_VENTA = "viviendas_venta.parquet"
OBJ_VIVIENDAS_ALQUILER = "viviendas_alquiler.parquet"


def cargar_datos():
    client = crear_cliente_minio()
    df_venta = bajar_minio(client, PATH_PRIMARIOS_LIMPIO, OBJ_VIVIENDAS_VENTA)
    df_alquiler = bajar_minio(client, PATH_PRIMARIOS_LIMPIO, OBJ_VIVIENDAS_ALQUILER)

    df_venta["tipo"] = "venta"
    df_alquiler["tipo"] = "alquiler"

    return pd.concat([df_venta, df_alquiler])


def agrupar_tipo(x):
    if x == "Particular":
        return "Particular"
    elif x in ["Agente Pro", "Profesional"]:
        return "Intermediario"
    elif x == "Promotora":
        return "Promotora"


def preparar_datos(df):
    df["grupo"] = df["Anuncia"].apply(agrupar_tipo)

    X = df["Descripcion"]
    y = df["grupo"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def entrenar_evaluar(vectorizer, X_train, y_train, X_eval, y_eval):
    spanish_stopwords = stopwords.words("spanish")

    model = Pipeline([
        ("vectorizer", vectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=spanish_stopwords
        )),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    acc = accuracy_score(y_eval, y_pred)
    f1_macro = f1_score(y_eval, y_pred, average="macro")
    recall_macro = recall_score(y_eval, y_pred, average="macro")
    precision_macro = precision_score(y_eval, y_pred, average="macro")

    print("Accuracy:", acc)
    print("F1 macro:", f1_macro)
    print("Recall macro:", recall_macro)
    print("Precision macro:", precision_macro)
    print(classification_report(y_eval, y_pred))

    return acc, f1_macro, recall_macro, precision_macro


def run_experiment(name, vectorizer, X_train, y_train, X_val, y_val):
    run = wandb.init(
        project="random-forest-texto",
        name=name,
        entity="pd1-c2526-team2",
        config={"vectorizer": name}
    )

    acc, f1_macro, recall_macro, precision_macro = entrenar_evaluar(
        vectorizer, X_train, y_train, X_val, y_val
    )

    wandb.log({
        "accuracy": acc,
        "f1_macro": f1_macro,
        "recall_macro": recall_macro,
        "precision_macro": precision_macro
    })

    run.finish()


def main():
    nltk.download("stopwords")

    df = cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_datos(df)

    wandb.login()

    run_experiment("CountVectorizer_RF", CountVectorizer, X_train, y_train, X_val, y_val)
    run_experiment("TfidfVectorizer_RF", TfidfVectorizer, X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()