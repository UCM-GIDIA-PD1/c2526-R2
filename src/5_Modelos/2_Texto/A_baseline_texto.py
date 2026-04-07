from funciones_texto import bajar_df_texto, x_y_split, train_val_test_split
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import wandb

def calculo_baseline_clase_mayoritaria(y_train, y_val):
    """
    Genera predicciones usando siempre la clase mayoritaria del conjunto de entrenamiento,
    evalúa el resultado sobre el conjunto de validación y sube las métricas a W&B.
    """
    print(f"Calculando Baseline...")

    # 1. Detectar clase mayoritaria en el conjunto de entrenamiento
    clase_mayoritaria = Counter(y_train).most_common(1)[0][0]

    # 2. Generar predicciones para el conjunto de validación
    y_pred_baseline = np.full(shape=len(y_val), fill_value=clase_mayoritaria)

    # 3. Calcular métrica principal (Accuracy)
    accuracy_base = accuracy_score(y_val, y_pred_baseline)

    # 4. Iniciar ejecución y registrar métricas en W&B
    run = wandb.init(
        entity="pd1-c2526-team2",
        project=f"modelo-texto",
        name="baseline-clase-mayoritaria",
        job_type="baseline-evaluation",
        group="texto_baseline",
    )

    wandb.log({
        "val_accuracy": accuracy_base,
        "clase_mayoritaria": clase_mayoritaria,
        "modelo": "Baseline (Clase Mayoritaria)"
    })

    run.finish()

    # 5. Generar reportes adicionales para la consola
    report = classification_report(y_val, y_pred_baseline, zero_division=0)
    matrix = confusion_matrix(y_val, y_pred_baseline)

    return clase_mayoritaria, accuracy_base, report, matrix


if __name__ == "__main__":
    # Descarga y partición de datos
    df = bajar_df_texto()
    x, y = x_y_split(df)
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y)

    # Ejecución del cálculo del baseline
    clase, acc, report, matrix = calculo_baseline_clase_mayoritaria(y_train, y_val)

    # Impresión de resultados en la terminal
    print("\n=== BASELINE: Clase mayoritaria ===")
    print(f"Clase predicha siempre: {clase}\n")
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(report)

    print("Confusion Matrix:")
    print(matrix)