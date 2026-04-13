import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb
from funciones_texto import bajar_df_texto, x_y_split, train_val_test_split

def calculo_baseline_clase_mayoritaria(
    y_train: pd.Series, 
    y_val: pd.Series, 
    y_test: pd.Series
) -> None:
    """
    Calcula y registra las métricas de un modelo baseline basado en la clase mayoritaria.

    Determina la clase más frecuente combinando train y val, genera una predicción 
    constante para el set de test y reporta métricas a Weights & Biases.

    Args:
        y_train (pd.Series): Etiquetas de entrenamiento.
        y_val (pd.Series): Etiquetas de validación.
        y_test (pd.Series): Etiquetas de prueba para evaluación final.
    """
    # Unimos train y val para determinar la clase mayoritaria (representa el conocimiento disponible)
    y_train_val: pd.Series = pd.concat([y_train, y_val])
    clase_mayoritaria = Counter(y_train_val).most_common(1)[0][0]

    # Predicción: vector constante con la clase mayoritaria
    y_pred_baseline: np.ndarray = np.full(shape=len(y_test), fill_value=clase_mayoritaria)

    # Cálculo de métricas
    metrics: dict[str, float] = {
        "test_f1_macro": float(f1_score(y_test, y_pred_baseline, average='macro')),
        "test_precision_macro": float(precision_score(y_test, y_pred_baseline, average='macro', zero_division=0)),
        "test_recall_macro": float(recall_score(y_test, y_pred_baseline, average='macro', zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_baseline))
    }

    # Logging en Weights & Biases usando context manager
    with wandb.init(
        entity="pd1-c2526-team2",
        project="modelo-texto",
        name="baseline-clase-mayoritaria",
        job_type="baseline-evaluation",
        group="texto_baseline",
        config={"model_type": "baseline"},
        reinit=True
    ) as run:
        wandb.log(metrics)
    
    print(f"Baseline calculado correctamente. F1-Macro: {metrics['test_f1_macro']:.4f}")

if __name__ == "__main__":
    # 1. Obtención de datos
    df = bajar_df_texto()
    x, y = x_y_split(df)
    
    # 2. Split (Obtenemos X e Y, pero solo usaremos Y para este baseline)
    _, _, _, y_train, y_val, y_test = train_val_test_split(x, y)
    
    # 3. Ejecución (Solo pasamos las etiquetas)
    calculo_baseline_clase_mayoritaria(y_train, y_val, y_test)