import sys
import os

# Añadir el directorio raíz del proyecto al path para poder importar utils
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# También añadir el directorio src al path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Añadir el propio directorio para importar EvaluacionRF_LR_LSTM
SELF_DIR = os.path.dirname(os.path.abspath(__file__))
if SELF_DIR not in sys.path:
    sys.path.insert(0, SELF_DIR)

# funciones_texto.py vive en 5_Modelos/2_Texto — añadirlo al path
FUNCIONES_TEXTO_DIR = os.path.abspath(os.path.join(SRC_DIR, "5_Modelos", "2_Texto"))
if FUNCIONES_TEXTO_DIR not in sys.path:
    sys.path.insert(0, FUNCIONES_TEXTO_DIR)

from EvaluacionRF_LR_LSTM import main

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  EVALUACIÓN DE MODELOS - TEXTO")
    print("=" * 50)
    print("  Ejecutando evaluación de: Logistic Regression, Random Forest y LSTM")
    print("=" * 50 + "\n")
    main()
