import pandas as pd
import numpy as np
import wandb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from utils.funciones_minio import crear_cliente_minio, bajar_minio

def entrenar_lasso(df, nombre_mercado):
    """
    Entrena un modelo Lasso con búsqueda de hiperparámetros (GridSearch)
    y registra los resultados en Weights & Biases.
    """
    # 1. Separación de variables
    X = df.drop(columns=['Precio'])
    y = df['Precio']

    # Separamos columnas que son números y que son texto/booleanos
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'str', 'bool']).columns.tolist()

    # Para las categorías, imputamos con "Desconocido"
    X[cat_cols] = X[cat_cols].fillna('Desconocido')
    X[cat_cols] = X[cat_cols].astype(str)

    # 2. Partición Estratificada (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=X['Distrito'])

    # 3. Preprocesamiento de variables
    # Los números se escalan y se imputan nulos con la mediana si los hay
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])

    # Las categorías se pasan a 0 y 1 (OneHotEncoding)
    cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    # Juntamos ambos preprocesamientos
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols),('cat', cat_transformer, cat_cols)])

   # 4. Definición del Modelo y GridSearch
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('regressor', Lasso(max_iter=50000, random_state=42))])

    # Valores de la penalización L1 a probar
    param_grid = {'regressor__alpha': [1.0, 10.0, 100.0, 500.0, 1000.0]}

    # Inicializamos W&B
    run = wandb.init(
        entity="pd1-c2526-team2",
        project=f"modelo-precio-viviendas-{nombre_mercado}",
        name=f"lasso-{nombre_mercado}",
        job_type="model-training",
        group=nombre_mercado,
        config={
            "model": "Lasso",
            "params_tried": param_grid,
            "test_size": 0.2,
            "random_state": 42
        }
    )

    print(f"Buscando mejor Alpha (5-Fold CV) para {nombre_mercado}...")
    
    # Usamos R2 como métrica para elegir el mejor Alpha durante la validación cruzada
    grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculamos las métricas finales
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "best_alpha": grid_search.best_params_['regressor__alpha']
    }

    # Subimos a W&B
    wandb.log(metrics)
    run.finish()

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_ventas_regresion.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_regresion.parquet")

    entrenar_lasso(df_venta, "venta")
    entrenar_lasso(df_alquiler, "alquiler")