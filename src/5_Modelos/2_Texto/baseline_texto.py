from funciones_texto import bajar_df_texto, separar_texto_train_test
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix




def baseline_clase_mayoritaria(y_train, y_test):
    """
    Genera predicciones usando siempre la clase mayoritaria del conjunto de entrenamiento
    y evalúa el resultado sobre el conjunto de test.
    """

    # Detectar clase mayoritaria
    clase_mayoritaria = Counter(y_train).most_common(1)[0][0]

    # Generar predicciones
    y_pred = np.full(shape=len(y_test), fill_value=clase_mayoritaria)

    # Evaluación
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return clase_mayoritaria, accuracy, report, matrix



if __name__ == "__main__":
    df = bajar_df_texto()
    X_train, X_test, y_train, y_test = separar_texto_train_test(df)

    clase, acc, report, matrix = baseline_clase_mayoritaria(y_train, y_test)

    print("=== BASELINE: Clase mayoritaria ===")
    print(f"Clase predicha siempre: {clase}\n")
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(report)

    print("Confusion Matrix:")
    print(matrix)