import wandb
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import nltk
from nltk.corpus import stopwords

from funciones_texto import bajar_df_texto, x_y_split, train_val_test_split


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
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
    }


def preparar_tokenizer(X_train, X_val, X_test, max_words, max_len):
    
    """Prepara y aplica un tokenizador sobre datos de texto.

    Ajusta un Tokenizer sobre el conjunto de entrenamiento y transforma
    los textos de entrenamiento, validación y test en secuencias numéricas
    con padding a una longitud fija.

    Args:
        X_train: Textos de entrenamiento.
        X_val: Textos de validación.
        X_test: Textos de test.
        max_words (int): Número máximo de palabras a considerar en el vocabulario.
        max_len (int): Longitud máxima de las secuencias.

    Returns:
        tuple:
            - X_train_seq: Secuencias de entrenamiento.
            - X_val_seq: Secuencias de validación.
            - X_test_seq: Secuencias de test.
    """

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
    X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=max_len)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

    return X_train_seq, X_val_seq, X_test_seq


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
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
        LSTM(lstm_units),
        Dense(3, activation="softmax")  # 3 clases
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


def entrenar_lstm_texto(X_train, y_train, X_val, y_val, X_test, y_test):

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
        X_val: Textos de validación.
        y_val: Etiquetas de validación.
        X_test: Textos de test.
        y_test: Etiquetas de test.

    Returns:
        tuple:
            - mejor_modelo: Modelo entrenado con la mejor configuración encontrada.
            - mejor_resultado (dict): Diccionario con la configuración y métricas del mejor modelo.
    """

    clases = {label: i for i, label in enumerate(sorted(y_train.unique()))}
    y_train_enc = y_train.map(clases).values
    y_val_enc = y_val.map(clases).values
    y_test_enc = y_test.map(clases).values

    max_words_list = [5000, 10000]
    max_len_list = [100, 200]
    embedding_dims = [50, 100]
    lstm_units_list = [64, 128]
    batch_sizes = [32]
    epochs_list = [3]  

    mejor_resultado = None
    mejor_modelo = None

    run = wandb.init(
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
    ])

    for max_words in max_words_list:
        for max_len in max_len_list:
            for emb_dim in embedding_dims:
                for lstm_units in lstm_units_list:

                    nombre = f"mw{max_words}_len{max_len}_emb{emb_dim}_lstm{lstm_units}"
                    print(f"\nEntrenando: {nombre}")

                    X_train_seq, X_val_seq, X_test_seq = preparar_tokenizer(
                        X_train, X_val, X_test, max_words, max_len
                    )

                    model = crear_modelo(max_words, max_len, emb_dim, lstm_units)

                    model.fit(
                        X_train_seq, y_train_enc,
                        validation_data=(X_val_seq, y_val_enc),
                        epochs=3,
                        batch_size=32,
                        verbose=0
                    )

                    y_pred_val = np.argmax(model.predict(X_val_seq), axis=1)

                    metricas = evaluar_modelo(y_val_enc, y_pred_val)

                    print(metricas)

                    table.add_data(
                        nombre,
                        max_words,
                        max_len,
                        emb_dim,
                        lstm_units,
                        metricas["f1_macro"],
                        metricas["accuracy"]
                    )

                    if (mejor_resultado is None) or (metricas["f1_macro"] > mejor_resultado["f1_macro"]):
                        mejor_resultado = metricas | {
                            "nombre": nombre,
                            "max_words": max_words,
                            "max_len": max_len,
                            "embedding_dim": emb_dim,
                            "lstm_units": lstm_units
                        }
                        mejor_modelo = model
                        best_tokenizer_data = (X_train_seq, X_val_seq, X_test_seq)

    print("\n=== MEJOR MODELO LSTM ===")
    print(mejor_resultado)

    wandb.log({"resultados_modelos": table})

    wandb.log({
        "f1_por_modelo": wandb.plot.bar(
            table, "modelo", "f1_macro", title="F1 por modelo (LSTM)"
        )
    })

    X_train_seq, X_val_seq, X_test_seq = best_tokenizer_data

    y_pred_test = np.argmax(mejor_modelo.predict(X_test_seq), axis=1)
    metricas_test = evaluar_modelo(y_test_enc, y_pred_test)

    print("\n=== RESULTADOS EN TEST ===")
    print(metricas_test)

    wandb.log({
        "test_f1": metricas_test["f1_macro"],
        "test_accuracy": metricas_test["accuracy"]
    })

    run.finish()

    return mejor_modelo, mejor_resultado


def main():
    
    """Función principal del script.

    Descarga los recursos necesarios de NLTK, carga y prepara el dataset de texto,
    divide los datos en entrenamiento, validación y test, inicia sesión en wandb
    y ejecuta el entrenamiento del modelo LSTM para clasificación de texto.
    """
    
    nltk.download("stopwords")

    df = bajar_df_texto()
    X, y = x_y_split(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    wandb.login()

    entrenar_lstm_texto(X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    main()