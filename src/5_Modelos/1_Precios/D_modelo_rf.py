import pandas as pd
import numpy as np
import wandb
import time   
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats

from utils.funciones_minio import crear_cliente_minio, bajar_minio

def entrenar_rf_tuning(df, nombre_mercado):
    print(f"BÚSQUEDA DE HIPERPARÁMETROS RANDOM FOREST: {nombre_mercado.upper()}")

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
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('regressor', RandomForestRegressor(random_state=42))])

    # Inicializamos W&B
    run = wandb.init(
        entity="pd1-c2526-team2",
        project=f"modelo-precio-viviendas-{nombre_mercado}",
        name=f"rf-tuning-{nombre_mercado}",
        job_type="hyperparameter-tuning",
        group=nombre_mercado
    )

    # Estrategia 1: GridSearchCV
    print("\n Estrategia 1: GridSearchCV")
    param_grid = {
        'regressor__n_estimators': [200, 400],
        'regressor__max_depth': [15, 25],
        'regressor__min_samples_split': [2, 5],
        'regressor__max_features': ['sqrt', 'log2', 1.0]
    }
    
    inicio_grid = time.time()
    grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    tiempo_grid = time.time() - inicio_grid
    
    print(f"   Mejores Params (Grid): {grid_search.best_params_}")
    print(f"   Mejor R2 CV (Grid): {grid_search.best_score_:.4f}")
    
    # Estrategia 2: Randomized Search
    print("\n Estrategia 2: RandomizedSearchCV")
    param_dist = {
        'regressor__n_estimators': stats.randint(100, 800),
        'regressor__max_depth': [None, 10, 20, 30, 40],
        'regressor__min_samples_split': stats.randint(2, 15),
        'regressor__min_samples_leaf': stats.randint(1, 5),
        'regressor__max_features': ['sqrt', 'log2', 1.0]
    }
    
    inicio_random = time.time()
    random_search = RandomizedSearchCV(full_pipeline, param_distributions=param_dist, n_iter=50, cv=5, scoring='r2', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    tiempo_random = time.time() - inicio_random

    print(f"   Mejores Params (Random): {random_search.best_params_}")
    print(f"   Mejor R2 CV (Random): {random_search.best_score_:.4f}")

    # Selección de la mejor estrategia
    if grid_search.best_score_ >= random_search.best_score_:
        print(f"\n Ganador: GridSearchCV")
        mejor_estrategia = "GridSearchCV"
        mejores_params = grid_search.best_params_
        mejor_r2 = grid_search.best_score_
    else:
        print(f"\n Ganador: RandomizedSearchCV")
        mejor_estrategia = "RandomizedSearchCV"
        mejores_params = random_search.best_params_
        mejor_r2 = random_search.best_score_
    
    # Subimos los resultados clave a W&B
    wandb.log({
        "mejor_estrategia": mejor_estrategia,
        "mejor_n_estimators": mejores_params['regressor__n_estimators'],
        "mejor_max_depth": mejores_params.get('regressor__max_depth', None), 
        "mejor_min_samples_split": mejores_params['regressor__min_samples_split'],
        "mejor_min_samples_leaf": mejores_params.get('regressor__min_samples_leaf', 1),
        "mejor_max_features": str(mejores_params.get('regressor__max_features', 1.0)),
        "val_r2_cv": mejor_r2,
        "tiempo_grid_segundos": tiempo_grid,
        "tiempo_random_segundos": tiempo_random,
        "mercado": nombre_mercado,
        "modelo": "RandomForest (Tuning)"
    })
    
    run.finish()
    print("\n Búsqueda terminada. Copia los hiperparámetros ganadores para el script de evaluación.")
    
if __name__ == "__main__":
    cliente = crear_cliente_minio()
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_ventas_arboles.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_arboles.parquet")
    
    entrenar_rf_tuning(df_venta, "venta")
    entrenar_rf_tuning(df_alquiler, "alquiler")