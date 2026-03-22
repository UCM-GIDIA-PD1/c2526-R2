import pandas as pd
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.funciones_minio import crear_cliente_minio, bajar_minio

def calculo_medias(df, nombre_mercado):
    """
    Calcula el baseline y lo sube a W&B diferenciando por mercado.
    """
    X = df.drop(columns=['Precio'])
    y = df['Precio']

    # Partición estratificada por distrito
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=X['Distrito']
    )

    print(f"Calculando Baseline de la Media para {nombre_mercado}...")
    media_entrenamiento = y_train.mean()

    # Vector de predicciones del baseline (la media del entrenamiento, para todos los casos)
    y_pred_baseline = np.full_like(y_test, fill_value=media_entrenamiento)

    # Cálculo de métricas
    mae_base = mean_absolute_error(y_test, y_pred_baseline)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    r2_base = r2_score(y_test, y_pred_baseline)

    run = wandb.init(
        entity="pd1-c2526-team2",
        project="modelo-precio-viviendas", 
        name=f"baseline-{nombre_mercado}",
        job_type="baseline-evaluation",
        group=nombre_mercado
    )

    # Subimos las métricas
    wandb.log({
        "mae": mae_base,
        "rmse": rmse_base,
        "r2": r2_base,
        "precio_medio_train": media_entrenamiento,
        "mercado": nombre_mercado
    })

    run.finish()

if __name__ == "__main__":
    df_venta = bajar_minio(crear_cliente_minio(), "dataset_ml/precios/ventas", "df_venta_limpio.parquet")
    df_alquiler = bajar_minio(crear_cliente_minio(), "dataset_ml/precios/alquiler", "df_alquiler_limpio.parquet")

    calculo_medias(df_venta, "venta")
    calculo_medias(df_alquiler, "alquiler")
    