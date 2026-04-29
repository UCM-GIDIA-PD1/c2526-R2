import wandb
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold

import nltk
from nltk.corpus import stopwords

from funciones_texto import bajar_df_texto, x_y_split, train_val_test_split

import optuna




def evaluar_modelo(y_true, y_pred):

    """Devuelve las métricas de evaluación seleccionadas

    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas por el modelo.

    Returns:
        dict: Diccionario con las métricas calculadas:
            - accuracy (float)
            - f1_macro (float)
            - recall_macro (float)
            - precision_macro (float)
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
    }


def crear_modelo(max_words, max_len, embedding_dim, lstm_units):

    """Crea y compila un modelo de red neuronal LSTM para clasificación de texto.

    El modelo consiste en una capa de embedding, seguida de una capa LSTM
    y una capa densa final con activación softmax para clasificación multiclase.

    Args:
        max_words (int): Tamaño del vocabulario.
        max_len (int): Longitud de entrada de las secuencias.
        embedding_dim (int): Dimensión del espacio de embeddings.
        lstm_units (int): Número de unidades en la capa LSTM.

    Returns:
        keras.Model: Modelo compilado listo para entrenar.
    """

    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim),
        LSTM(lstm_units),
        Dense(3, activation="softmax")  # 3 clases
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


def entrenar_lstm_texto(X_train, y_train, X_test, y_test):

    """Entrena y selecciona el mejor modelo LSTM para clasificación de texto.

    Realiza una búsqueda de hiperparámetros sobre:
    - Tamaño del vocabulario (max_words)
    - Longitud de secuencia (max_len)
    - Dimensión de embeddings
    - Número de unidades LSTM

    Para cada configuración:
    - Prepara los datos mediante tokenización y padding
    - Entrena el modelo LSTM
    - Evalúa en el conjunto de validación
    - Registra los resultados en Weights & Biases (wandb)

    El mejor modelo según F1 macro se evalúa finalmente en el conjunto de test.

    Args:
        X_train: Textos de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        X_test: Textos de test.
        y_test: Etiquetas de test.

    Returns:
        tuple:
            - mejor_modelo: Modelo entrenado con la mejor configuración encontrada.
            - mejor_resultado (dict): Diccionario con la configuración y métricas del mejor modelo.
    """

    clases = {label: i for i, label in enumerate(sorted(y_train.unique()))}
    y_train_enc = y_train.map(clases).values
    y_test_enc = y_test.map(clases).values

    max_words_list = [5000, 10000]
    max_len_list = [100, 200]
    embedding_dims = [50, 100]
    lstm_units_list = [64, 128]
    batch_sizes = [32]
    epochs_list = [3]  

    mejor_resultado = None
    mejor_modelo = None

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    """run = wandb.init(
        entity="pd1-c2526-team2",
        project="modelo-texto-lstm",
        name="lstm-texto",
        config={
            "modelo": "LSTM",
            "task": "clasificacion",
        }
    )

    table = wandb.Table(columns=[
        "modelo", "max_words", "max_len", "embedding_dim", "lstm_units",
        "f1_macro", "accuracy"
    ])"""

    for max_words in max_words_list:
        for max_len in max_len_list:
            for emb_dim in embedding_dims:
                for lstm_units in lstm_units_list:

                    nombre = f"mw{max_words}_len{max_len}_emb{emb_dim}_lstm{lstm_units}"
                    print(f"\nEntrenando: {nombre}")

                    cv_scores = []

                    for train_idx, val_idx in skf.split(X_train, y_train_enc):

                        X_tr = X_train.iloc[train_idx]
                        X_vl = X_train.iloc[val_idx]
                        y_tr = y_train_enc[train_idx]
                        y_vl = y_train_enc[val_idx]

                        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
                        tokenizer.fit_on_texts(X_tr)

                        X_tr_seq = pad_sequences(tokenizer.texts_to_sequences(X_tr), maxlen=max_len)
                        X_vl_seq = pad_sequences(tokenizer.texts_to_sequences(X_vl), maxlen=max_len)

                        model = crear_modelo(max_words, max_len, emb_dim, lstm_units)

                        model.fit(
                            X_tr_seq, y_tr,
                            epochs=3,
                            batch_size=32,
                            verbose=0
                        )

                        y_pred_val = np.argmax(model.predict(X_vl_seq, verbose=0), axis=1)

                        f1 = f1_score(y_vl, y_pred_val, average="macro")
                        cv_scores.append(f1)

                    f1_mean = np.mean(cv_scores)
                    f1_std = np.std(cv_scores)

                    print({"f1_macro": f1_mean, "f1_std": f1_std})

                    """table.add_data(
                        nombre,
                        max_words,
                        max_len,
                        emb_dim,
                        lstm_units,
                        metricas["f1_macro"],
                        metricas["accuracy"]
                    )"""

                    if (mejor_resultado is None) or (f1_mean > mejor_resultado["f1_macro"]):
                        mejor_resultado = {
                            "nombre": nombre,
                            "max_words": max_words,
                            "max_len": max_len,
                            "embedding_dim": emb_dim,
                            "lstm_units": lstm_units,
                            "f1_macro": f1_mean,
                            "f1_std": f1_std
                        }
                        mejor_config = (max_words, max_len, emb_dim, lstm_units)

    print("\n=== MEJOR MODELO LSTM ===")
    print(mejor_resultado)

    """wandb.log({"resultados_modelos": table})

    wandb.log({
        "f1_por_modelo": wandb.plot.bar(
            table, "modelo", "f1_macro", title="F1 por modelo (LSTM)"
        )
    })"""

    max_words, max_len, emb_dim, lstm_units = mejor_config

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

    mejor_modelo = crear_modelo(max_words, max_len, emb_dim, lstm_units)

    mejor_modelo.fit(
        X_train_seq, y_train_enc,
        epochs=3,
        batch_size=32,
        verbose=0
    )

    y_pred_test = np.argmax(mejor_modelo.predict(X_test_seq), axis=1)
    metricas_test = evaluar_modelo(y_test_enc, y_pred_test)

    print("\n=== RESULTADOS EN TEST ===")
    print(metricas_test)
    print("\n=== ANALISIS POR LONGITUD ===")
    print(analizar_por_longitud(X_test, y_test_enc, y_pred_test))

    """wandb.log({
        "test_f1": metricas_test["f1_macro"],
        "test_accuracy": metricas_test["accuracy"]
    })

    run.finish()"""

    return mejor_modelo, mejor_resultado

def analizar_por_longitud(X_text, y_true, y_pred):

    import pandas as pd

    df = pd.DataFrame({
        "texto": X_text,
        "y_true": y_true,
        "y_pred": y_pred
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

        f1 = f1_score(subset["y_true"], subset["y_pred"], average="macro")

        resultados.append({
            "segmento": seg,
            "n_samples": len(subset),
            "f1_macro": f1
        })

    return pd.DataFrame(resultados)

def entrenar_lstm_texto_optuna(X_train, y_train, X_test, y_test):

    clases = {label: i for i, label in enumerate(sorted(y_train.unique()))}
    y_train_enc = y_train.map(clases).values
    y_test_enc = y_test.map(clases).values

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):

        max_words = trial.suggest_categorical("max_words", [3000, 5000, 8000, 12000, 15000])
        max_len = trial.suggest_int("max_len", 80, 250, step=40)
        embedding_dim = trial.suggest_categorical("embedding_dim", [50, 100, 150])
        lstm_units = trial.suggest_int("lstm_units", 32, 128, step=32)
        epochs = trial.suggest_int("epochs", 2, 5)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_enc)):

            X_tr = X_train.iloc[train_idx]
            X_vl = X_train.iloc[val_idx]
            y_tr = y_train_enc[train_idx]
            y_vl = y_train_enc[val_idx]

            tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
            tokenizer.fit_on_texts(X_tr)

            X_tr_seq = pad_sequences(tokenizer.texts_to_sequences(X_tr), maxlen=max_len)
            X_vl_seq = pad_sequences(tokenizer.texts_to_sequences(X_vl), maxlen=max_len)

            model = crear_modelo(max_words, max_len, embedding_dim, lstm_units)

            model.fit(
                X_tr_seq, y_tr,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )

            y_pred = np.argmax(model.predict(X_vl_seq, verbose=0), axis=1)

            f1 = f1_score(y_vl, y_pred, average="macro")
            cv_scores.append(f1)

            trial.report(f1, step=fold_idx)

            if trial.should_prune():
                K.clear_session()
                raise optuna.TrialPruned()
            
            K.clear_session()

        return np.mean(cv_scores)

    print("Buscando mejor modelo con Optuna...")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
    )

    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    print("\n=== MEJOR CONFIG OPTUNA ===")
    print(best_params)

    max_words = best_params["max_words"]
    max_len = best_params["max_len"]
    embedding_dim = best_params["embedding_dim"]
    lstm_units = best_params["lstm_units"]
    epochs = best_params["epochs"]
    batch_size = best_params["batch_size"]

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

    mejor_modelo = crear_modelo(max_words, max_len, embedding_dim, lstm_units)

    mejor_modelo.fit(
        X_train_seq, y_train_enc,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    y_pred_test = np.argmax(mejor_modelo.predict(X_test_seq), axis=1)
    metricas_test = evaluar_modelo(y_test_enc, y_pred_test)

    print("\n=== RESULTADOS EN TEST ===")
    print(metricas_test)

    print("\n=== ANALISIS POR LONGITUD ===")
    print(analizar_por_longitud(X_test, y_test_enc, y_pred_test))

    return mejor_modelo, best_params


def main():
    
    """Función principal del script.

    Descarga los recursos necesarios de NLTK, carga y prepara el dataset de texto,
    divide los datos en entrenamiento, validación y test, inicia sesión en wandb
    y ejecuta el entrenamiento del modelo LSTM para clasificación de texto.
    """
    
    nltk.download("stopwords")

    df = bajar_df_texto()
    X, y = x_y_split(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    #wandb.login()

    modo = input("Selecciona modo: 1 (Grid) / 2 (Optuna): ")

    if modo == "1":
        entrenar_lstm_texto(X_train, y_train, X_test, y_test)
    elif modo == "2":
        entrenar_lstm_texto_optuna(X_train, y_train, X_test, y_test)
    else:
        print("Opción no válida")
        entrenar_lstm_texto(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()