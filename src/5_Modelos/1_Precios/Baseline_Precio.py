import pandas as pd
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from utils.funciones_minio import crear_cliente_minio, bajar_minio

def calculo_medias_por_distrito(df, nombre_mercado):
    """
    Calcula el baseline usando la media de cada distrito y lo sube a W&B.
    """
    X = df.drop(columns=['Precio'])
    y = df['Precio']

    # Partición estratificada por distrito
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=X['Distrito']
    )

    print(f"Calculando Baseline por Distrito para {nombre_mercado}...")

    # 1. Calculamos la media de precio POR DISTRITO
    medias_distrito_train = y_train.groupby(X_train['Distrito']).mean()

    # 2. Para cada piso del Test, miramos su distrito y le asignamos la media correspondiente
    y_pred_baseline = X_test['Distrito'].map(medias_distrito_train)

    # Por si algún distrito del test no estuviera en el train, rellenamos los nulos con la media general.
    y_pred_baseline = y_pred_baseline.fillna(y_train.mean())

    # 3. Calculamos las métricas comparando el precio real vs la media de su distrito
    mae_base = mean_absolute_error(y_test, y_pred_baseline)
    rmse_base = root_mean_squared_error(y_test, y_pred_baseline)
    r2_base = r2_score(y_test, y_pred_baseline)

    # 4. Registramos en W&B como UN SOLO experimento para poder compararlo luego con Lasso
    run = wandb.init(
        entity="pd1-c2526-team2",
        project="modelo-precio-viviendas", 
        name=f"baseline-distrito-{nombre_mercado}",
        job_type="baseline-evaluation",
        group=nombre_mercado,
    )

    wandb.log({
        "mae": mae_base,
        "rmse": rmse_base,
        "r2": r2_base,
        "mercado": nombre_mercado
    })

    run.finish()

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_venta_limpio.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_limpio.parquet")

    calculo_medias_por_distrito(df_venta, "venta")
    calculo_medias_por_distrito(df_alquiler, "alquiler")