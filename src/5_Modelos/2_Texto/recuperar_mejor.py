import wandb
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import nltk
from funciones_texto import bajar_df_texto, x_y_split

import optuna


# =========================
# MÉTRICAS
# =========================
def evaluar_modelo(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
    }


# =========================
# MODELO
# =========================
def crear_modelo(max_words, embedding_dim, lstm_units):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim),
        LSTM(lstm_units),
        Dense(3, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


# =========================
# GRID SEARCH
# =========================
def entrenar_lstm_texto(X_train, y_train, X_test, y_test):

    run = wandb.init(
        project="modelo-texto-f",
        entity="pd1-c2526-team2",
        name="lstm-grid",
        group="lstm-grid",
        job_type="grid",
        config={"model": "lstm", "search": "grid"}
    )

    clases = {label: i for i, label in enumerate(sorted(y_train.unique()))}
    y_train_enc = y_train.map(clases).values
    y_test_enc = y_test.map(clases).values

    max_words_list = [5000, 10000]
    max_len_list = [100, 200]
    embedding_dims = [50, 100]
    lstm_units_list = [64, 128]

    mejor_resultado = None
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for max_words in max_words_list:
        for max_len in max_len_list:
            for emb_dim in embedding_dims:
                for lstm_units in lstm_units_list:

                    nombre = f"mw{max_words}_len{max_len}_emb{emb_dim}_lstm{lstm_units}"

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

                        model = crear_modelo(max_words, emb_dim, lstm_units)

                        model.fit(X_tr_seq, y_tr, epochs=3, batch_size=32, verbose=0)

                        y_pred = np.argmax(model.predict(X_vl_seq, verbose=0), axis=1)
                        cv_scores.append(f1_score(y_vl, y_pred, average="macro"))

                        K.clear_session()

                    f1_mean = np.mean(cv_scores)
                    f1_std = np.std(cv_scores)

                    wandb.log({
                        "model_name": nombre,
                        "cv_f1_mean": f1_mean,
                        "cv_f1_std": f1_std,
                        "max_words": max_words,
                        "max_len": max_len,
                        "embedding_dim": emb_dim,
                        "lstm_units": lstm_units
                    })

                    if mejor_resultado is None or f1_mean > mejor_resultado["f1_macro"]:
                        mejor_resultado = {
                            "nombre": nombre,
                            "max_words": max_words,
                            "max_len": max_len,
                            "embedding_dim": emb_dim,
                            "lstm_units": lstm_units,
                            "f1_macro": f1_mean
                        }

                        mejor_config = (max_words, max_len, emb_dim, lstm_units)

    wandb.log({
        "best_cv_f1": mejor_resultado["f1_macro"],
        "best_model": mejor_resultado["nombre"]
    })

    max_words, max_len, emb_dim, lstm_units = mejor_config

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

    modelo = crear_modelo(max_words, emb_dim, lstm_units)
    modelo.fit(X_train_seq, y_train_enc, epochs=3, batch_size=32, verbose=0)

    y_pred = np.argmax(modelo.predict(X_test_seq), axis=1)
    metricas = evaluar_modelo(y_test_enc, y_pred)

    wandb.log({
        "test_f1_macro": metricas["f1_macro"],
        "test_accuracy": metricas["accuracy"],
        "test_precision": metricas["precision_macro"],
        "test_recall": metricas["recall_macro"]
    })

    modelo.save("lstm_grid_best.h5")
    wandb.save("lstm_grid_best.h5")

    run.finish()

    return modelo, mejor_resultado


# =========================
# OPTUNA
# =========================
def entrenar_lstm_texto_optuna(X_train, y_train, X_test, y_test):

    run = wandb.init(
        project="modelo-texto-f",
        entity="pd1-c2526-team2",
        name="lstm-optuna",
        group="lstm-optuna",
        job_type="optuna",
        config={"model": "lstm", "search": "optuna"}
    )

    clases = {label: i for i, label in enumerate(sorted(y_train.unique()))}
    y_train_enc = y_train.map(clases).values
    y_test_enc = y_test.map(clases).values

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):

        max_words = trial.suggest_categorical("max_words", [3000, 5000, 10000])
        max_len = trial.suggest_int("max_len", 80, 200, step=40)
        embedding_dim = trial.suggest_categorical("embedding_dim", [50, 100])
        lstm_units = trial.suggest_int("lstm_units", 32, 128, step=32)
        epochs = trial.suggest_int("epochs", 2, 4)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])

        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_enc)):

            X_tr = X_train.iloc[train_idx]
            X_vl = X_train.iloc[val_idx]
            y_tr = y_train_enc[train_idx]
            y_vl = y_train_enc[val_idx]

            tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
            tokenizer.fit_on_texts(X_tr)

            X_tr_seq = pad_sequences(tokenizer.texts_to_sequences(X_tr), maxlen=max_len)
            X_vl_seq = pad_sequences(tokenizer.texts_to_sequences(X_vl), maxlen=max_len)

            model = crear_modelo(max_words, embedding_dim, lstm_units)

            model.fit(X_tr_seq, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)

            y_pred = np.argmax(model.predict(X_vl_seq, verbose=0), axis=1)
            f1 = f1_score(y_vl, y_pred, average="macro")

            scores.append(f1)

            wandb.log({"trial_f1": f1, "fold": fold_idx})

            trial.report(f1, step=fold_idx)

            if trial.should_prune():
                K.clear_session()
                raise optuna.TrialPruned()

            K.clear_session()

        mean_f1 = np.mean(scores)
        wandb.log({"trial_cv_f1_mean": mean_f1})

        return mean_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    wandb.log({
        "best_params": best_params,
        "best_cv_f1": study.best_value
    })

    tokenizer = Tokenizer(num_words=best_params["max_words"], oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=best_params["max_len"])
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=best_params["max_len"])

    model = crear_modelo(
        best_params["max_words"],
        best_params["embedding_dim"],
        best_params["lstm_units"]
    )

    model.fit(
        X_train_seq, y_train_enc,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        verbose=0
    )

    y_pred = np.argmax(model.predict(X_test_seq), axis=1)
    metricas = evaluar_modelo(y_test_enc, y_pred)

    wandb.log({
        "test_f1_macro": metricas["f1_macro"],
        "test_accuracy": metricas["accuracy"],
        "test_precision": metricas["precision_macro"],
        "test_recall": metricas["recall_macro"]
    })

    model.save("lstm_optuna_best.h5")
    wandb.save("lstm_optuna_best.h5")

    run.finish()

    return model, best_params


# =========================
# MAIN
# =========================
def main():

    nltk.download("stopwords")

    df = bajar_df_texto()
    X, y = x_y_split(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    modo = input("Selecciona modo: 1 (Grid) / 2 (Optuna): ")

    if modo == "1":
        entrenar_lstm_texto(X_train, y_train, X_test, y_test)
    else:
        entrenar_lstm_texto_optuna(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()