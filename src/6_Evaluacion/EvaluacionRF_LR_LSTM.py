import wandb
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.corpus import stopwords

from funciones_texto import bajar_df_texto, x_y_split, train_val_test_split


def evaluar(y_true, y_pred):
    return {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_f1_macro": f1_score(y_true, y_pred, average="macro"),
        "test_recall_macro": recall_score(y_true, y_pred, average="macro"),
        "test_precision_macro": precision_score(y_true, y_pred, average="macro"),
    }


def cargar_datos():
    df = bajar_df_texto()
    X, y = x_y_split(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # train final
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    return X_train_full, X_test, y_train_full, y_test


def run_logreg(X_train, X_test, y_train, y_test, stop_words):

    run = wandb.init(
        project="modelo-texto",
        name="LogReg_best",
        entity="pd1-c2526-team2",
        reinit=True
    )

    model = Pipeline([
        ("vectorizer", CountVectorizer(
            max_features=10000,
            ngram_range=(1,1),
            stop_words=stop_words
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    wandb.log(evaluar(y_test, y_pred))
    run.finish()


def run_rf(X_train, X_test, y_train, y_test, stop_words):

    run = wandb.init(
        project="modelo-texto",
        name="RandomForest_best",
        entity="pd1-c2526-team2",
        reinit=True
    )

    model = Pipeline([
        ("vectorizer", TfidfVectorizer(
            max_features=10000,
            stop_words=stop_words
        )),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    wandb.log(evaluar(y_test, y_pred))
    run.finish()


def run_lstm(X_train, X_test, y_train, y_test):

    run = wandb.init(
        project="modelo-texto",
        name="LSTM_best",
        entity="pd1-c2526-team2",
        reinit=True
    )

    clases = {label: i for i, label in enumerate(sorted(y_train.unique()))}
    y_train_enc = y_train.map(clases).values
    y_test_enc = y_test.map(clases).values

    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    max_len = 100

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

    model = Sequential([
        Embedding(input_dim=10000, output_dim=100, input_length=max_len),
        LSTM(64),
        Dense(3, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.fit(X_train_seq, y_train_enc, epochs=3, batch_size=32, verbose=0)

    y_pred = np.argmax(model.predict(X_test_seq), axis=1)

    wandb.log(evaluar(y_test_enc, y_pred))
    run.finish()


def main():
    nltk.download("stopwords")
    stop_words = stopwords.words("spanish")

    wandb.login()

    X_train, X_test, y_train, y_test = cargar_datos()

    run_logreg(X_train, X_test, y_train, y_test, stop_words)
    run_rf(X_train, X_test, y_train, y_test, stop_words)
    run_lstm(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()