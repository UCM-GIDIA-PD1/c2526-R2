import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb

from funciones_texto import (
    bajar_df_texto, x_y_split, train_val_test_split, obtener_o_cargar_embeddings
)


def optimizar_evaluar_naive_bayes(
    emb_train: np.ndarray,
    y_train: pd.Series,
    emb_val: np.ndarray,
    y_val: pd.Series,
    emb_test: np.ndarray,
    y_test: pd.Series,
) -> GaussianNB:
    """Busca los mejores hiperparámetros para Gaussian Naive Bayes sobre embeddings
    y evalúa el mejor modelo en el conjunto de test.

    La búsqueda respeta el split original (usa PredefinedSplit para evitar data leakage).

    Args:
        emb_train: Embeddings del conjunto de entrenamiento, shape (n_train, 384).
        y_train:   Etiquetas de entrenamiento.
        emb_val:   Embeddings del conjunto de validación, shape (n_val, 384).
        y_val:     Etiquetas de validación.
        emb_test:  Embeddings del conjunto de test, shape (n_test, 384).
        y_test:    Etiquetas de test.

    Returns:
        GaussianNB: El mejor modelo entrenado.
    """
    print("Configurando Gaussian Naive Bayes con embeddings...")

    # 1. Unir train+val para GridSearchCV con PredefinedSplit
    emb_train_val = np.vstack([emb_train, emb_val])
    y_train_val = pd.concat([y_train, y_val])

    # -1 → train (no se valida), 0 → val (se valida)
    test_fold = np.concatenate([
        np.full(len(emb_train), -1),
        np.full(len(emb_val), 0)
    ])
    ps = PredefinedSplit(test_fold)

    # 2. Grid de hiperparámetros
    # var_smoothing controla la estabilidad numérica (evita varianza cero)
    param_grid = {
        "var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3]
    }

    print("\nIniciando búsqueda de hiperparámetros...")
    grid_search = GridSearchCV(
        estimator=GaussianNB(),
        param_grid=param_grid,
        cv=ps,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(emb_train_val, y_train_val)

    best_model = grid_search.best_estimator_
    print(f"\nMejores hiperparámetros: {grid_search.best_params_}")

    # 3. Evaluar en test
    print("\nEvaluando el mejor modelo en el conjunto de TEST...")
    y_pred_test = best_model.predict(emb_test)

    metrics: dict[str, float] = {
        "test_f1_macro": float(f1_score(y_test, y_pred_test, average="macro")),
        "test_precision_macro": float(precision_score(y_test, y_pred_test, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test, y_pred_test, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
    }

    # 4. Logging en W&B
    with wandb.init(
        entity="pd1-c2526-team2",
        project="modelo-texto",
        name="naive-bayes-embeddings",
        job_type="hyperparameter-search-and-eval",
        group="texto_modelos_embeddings",
        config={
            "model_type": "GaussianNB",
            "vectorizer": "sentence-transformers",
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "search_strategy": "GridSearchCV",
            "best_params": grid_search.best_params_,
        },
        reinit=True,
    ):
        wandb.log(metrics)

    print(f"\nF1-Macro en Test: {metrics['test_f1_macro']:.4f}")
    return best_model


if __name__ == "__main__":
    df = bajar_df_texto()
    x, y = x_y_split(df)
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y)

    # Pre-computar embeddings UNA sola vez para los tres splits
    emb_train, emb_val, emb_test = obtener_o_cargar_embeddings(x_train, x_val, x_test)

    optimizar_evaluar_naive_bayes(emb_train, y_train, emb_val, y_val, emb_test, y_test)