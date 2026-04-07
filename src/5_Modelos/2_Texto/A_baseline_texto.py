import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb
from funciones_texto import bajar_df_texto, x_y_split, train_val_test_split

def calculo_baseline_clase_mayoritaria(x_train, y_train, x_val, y_val, x_test, y_test):
    # Unimos train y val para el baseline (90% de los datos)
    y_train_val = pd.concat([y_train, y_val])
    clase_mayoritaria = Counter(y_train_val).most_common(1)[0][0]

    # Evaluación sobre TEST
    y_pred_baseline = np.full(shape=len(y_test), fill_value=clase_mayoritaria)

    # Métricas
    metrics = {
        "test_f1_macro": f1_score(y_test, y_pred_baseline, average='macro'),
        "test_precision_macro": precision_score(y_test, y_pred_baseline, average='macro', zero_division=0),
        "test_recall_macro": recall_score(y_test, y_pred_baseline, average='macro', zero_division=0),
        "test_accuracy": accuracy_score(y_test, y_pred_baseline)
    }

    run = wandb.init(
        entity="pd1-c2526-team2",
        project="modelo-texto",
        name="baseline-clase-mayoritaria",
        job_type="baseline-evaluation",
        group="texto_baseline",
        config={"model_type": "baseline"}
    )

    wandb.log(metrics)
    run.finish()
    
    print(f"Baseline calculado. F1-Macro: {metrics['test_f1_macro']:.4f}")

if __name__ == "__main__":
    df = bajar_df_texto()
    x, y = x_y_split(df)
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y)
    calculo_baseline_clase_mayoritaria(x_train, y_train, x_val, y_val, x_test, y_test)