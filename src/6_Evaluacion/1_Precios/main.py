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

from utils.funciones_minio import crear_cliente_minio, bajar_minio

MODELOS = {
    "1": ("KNN",     "A_evaluacion_knn"),
    "2": ("Lasso",   "B_evaluacion_lasso"),
    "3": ("Random Forest", "C_evaluacion_rf"),
    "4": ("XGBoost", "D_evaluacion_xgboost"),
    "5": ("Subida a W&B (Modelo de Producción)", "E_entrenar_xgboost_produccion"),
}

def menu_precios():
    while True:
        print("\n" + "=" * 50)
        print("     EVALUACIÓN DE MODELOS - PRECIOS")
        print("=" * 50)
        for key, (nombre, _) in MODELOS.items():
            if key == "5":
                print("-" * 50)
            print(f"  {key}. Evaluar modelo: {nombre}")
        print("-" * 50)
        print("  0. Volver al menú principal")
        print("=" * 50)

        opcion = input("  Elige el modelo a evaluar: ").strip()

        if opcion == "0":
            print("\n  Volviendo al menú principal...")
            return

        if opcion not in MODELOS:
            print(f"\n  Error: Opción no válida. Elige entre 0 y {len(MODELOS)}.")
            continue

        nombre_modelo, modulo = MODELOS[opcion]
        print(f"\n  Cargando datos desde MinIO para evaluar {nombre_modelo}...")

        try:
            cliente = crear_cliente_minio()

            if modulo == "E_entrenar_xgboost_produccion":
                df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_venta_xgboost.parquet")
                df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_xgboost.parquet")
                
                import importlib
                mod = importlib.import_module(f"src.6_Evaluacion.1_Precios.{modulo}")
                mod.entrenar_y_guardar_produccion(df_venta, "venta")
                mod.entrenar_y_guardar_produccion(df_alquiler, "alquiler")
                continue

            if modulo == "B_evaluacion_lasso":
                df_venta    = bajar_minio(cliente, "dataset_ml/precios/ventas",   "df_ventas_regresion.parquet")
                df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_regresion.parquet")
            else:  # KNN, RF, XGBoost usan los parquets de árboles
                df_venta    = bajar_minio(cliente, "dataset_ml/precios/ventas",   "df_ventas_arboles.parquet")
                df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_arboles.parquet")

            # Importación dinámica del módulo de evaluación
            import importlib
            # Intentar importación relativa desde el paquete o por el path
            try:
                mod = importlib.import_module(f"src.6_Evaluacion.1_Precios.{modulo}")
            except ModuleNotFoundError:
                mod = importlib.import_module(modulo)

            if modulo == "A_evaluacion_knn":
                mod.evaluar_knn_final(df_venta,    "venta")
                mod.evaluar_knn_final(df_alquiler, "alquiler")
            elif modulo == "B_evaluacion_lasso":
                mod.evaluar_modelo_final(df_venta,    "venta")
                mod.evaluar_modelo_final(df_alquiler, "alquiler")
            elif modulo == "C_evaluacion_rf":
                mod.evaluar_rf_final(df_venta,    "venta")
                mod.evaluar_rf_final(df_alquiler, "alquiler")
            elif modulo == "D_evaluacion_xgboost":
                mod.evaluar_xgb_final_hibrido(df_venta,    "venta")
                mod.evaluar_xgb_final_hibrido(df_alquiler, "alquiler")

        except Exception as e:
            print(f"\n  Error durante la evaluación: {e}")


if __name__ == "__main__":
    menu_precios()
