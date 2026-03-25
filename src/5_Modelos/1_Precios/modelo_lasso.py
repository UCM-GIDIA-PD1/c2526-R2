import pandas as pd
import numpy as np
import wandb
import time   
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Lasso

from utils.funciones_minio import crear_cliente_minio, bajar_minio

def entrenar_lasso(df, nombre_mercado):
    """
    Entrena un modelo Lasso probando 2 estrategias (GridSearch y RandomizedSearch).
    Registra el mejor hiperparámetro en W&B para usarlo luego en la evaluación.
    """
    print(f"Entrenando Lasso para el mercado de {nombre_mercado}...")
    # 1. Separación de variables
    X = df.drop(columns=['Precio'])
    y = df['Precio']

    cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X[cat_cols] = X[cat_cols].fillna('Desconocido')
    X[cat_cols] = X[cat_cols].astype(str)

    # 2. Partición Estratificada (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=X['Distrito']
    )

    # 3. Preprocesamiento de variables
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])

    cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols),('cat', cat_transformer, cat_cols)])

    # 4. Definición del Pipeline Base
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('regressor', Lasso(max_iter=50000, random_state=42))])

    # Inicializamos W&B
    run = wandb.init(
        entity="pd1-c2526-team2",
        project=f"modelo-precio-viviendas-{nombre_mercado}",
        name=f"lasso-tuning-{nombre_mercado}",
        job_type="hyperparameter-tuning",
        group=nombre_mercado
    )

    # Estrategia 1: GridSearchCV
    print("\n Estrategia 1: GridSearchCV")
    param_grid = {'regressor__alpha': [1.0, 10.0, 100.0, 500.0, 1000.0]}
    
    inicio_grid = time.time()
    grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    tiempo_grid = time.time() - inicio_grid

    alpha_grid = grid_search.best_params_['regressor__alpha']
    print(f"   Mejor Alpha (Grid): {alpha_grid}")
    print(f"   Mejor R2 CV (Grid): {grid_search.best_score_:.4f}")

    # Estrategia 2: Randomized Search
    print("\n Estrategia 2: RandomizedSearchCV")
    import scipy.stats as stats
    param_dist = {'regressor__alpha': stats.uniform(1.0, 1000.0)} 
    
    inicio_random = time.time()
    random_search = RandomizedSearchCV(full_pipeline, param_distributions=param_dist, n_iter=5, cv=5, scoring='r2', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    tiempo_random = time.time() - inicio_random

    alpha_random = random_search.best_params_['regressor__alpha']
    print(f"   Mejor Alpha (Random): {alpha_random:.2f}")
    print(f"   Mejor R2 CV (Random): {random_search.best_score_:.4f}")

    # Seleccionamos la mejor estrategia comparando los mejores R2 obtenidos en CV
    if grid_search.best_score_ >= random_search.best_score_:
        print(f"\n Ganador: GridSearchCV con Alpha = {alpha_grid}")
        mejor_estrategia = "GridSearchCV"
        mejor_alpha = alpha_grid
        mejor_r2 = grid_search.best_score_
    else:
        print(f"\n Ganador: RandomizedSearchCV con Alpha = {alpha_random:.2f}")
        mejor_estrategia = "RandomizedSearchCV"
        mejor_alpha = alpha_random
        mejor_r2 = random_search.best_score_

    # Subimos los resultados clave a W&B para copiarlos luego
    wandb.log({
        "mejor_estrategia": mejor_estrategia,
        "mejor_alpha_final": mejor_alpha,
        "val_r2_cv": mejor_r2,  # Le llamamos val_r2_cv para saber que es de validación
        "tiempo_grid_segundos": tiempo_grid,
        "tiempo_random_segundos": tiempo_random,
        "mercado": nombre_mercado,
        "modelo": "Lasso (Tuning)"
    })
    
    run.finish()
    print("\nBúsqueda terminada. Revisa W&B para copiar el mejor_alpha_final.")

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_ventas_regresion.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_regresion.parquet")

    entrenar_lasso(df_venta, "venta")
    entrenar_lasso(df_alquiler, "alquiler")