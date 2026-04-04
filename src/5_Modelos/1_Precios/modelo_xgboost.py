import pandas as pd
import numpy as np
import wandb
import time   
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
import scipy.stats as stats

from utils.funciones_minio import crear_cliente_minio, bajar_minio

def entrenar_xgboost_tuning(df, nombre_mercado):
    print(f"BÚSQUEDA DE HIPERPARÁMETROS XGBOOST: {nombre_mercado.upper()}")

    # 1. Separación de variables (Precio Total directo)
    X = df.drop(columns=['Precio'])
    y = df['Precio'] 

    cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X[cat_cols] = X[cat_cols].fillna('Desconocido')
    X[cat_cols] = X[cat_cols].astype(str)

    # 2. Partición Estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=X['Distrito']
    )

    # 3. Preprocesamiento
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols),('cat', cat_transformer, cat_cols)])

    # 4. Pipeline con XGBoost
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('regressor', TransformedTargetRegressor(regressor=XGBRegressor(objective="reg:absoluteerror",random_state=42,n_jobs=-1),func=np.log1p,inverse_func=np.expm1))])

    run = wandb.init(
        entity="pd1-c2526-team2",
        project=f"modelo-precio-viviendas-{nombre_mercado}",
        name=f"xgb-tuning-{nombre_mercado}",
        job_type="hyperparameter-tuning",
        group=nombre_mercado
    )

    # Estrategia 1: GridSearch, probamos pocos hiperparámetros para no tardar demasiado
    print("\n Estrategia 1: GridSearchCV")
    param_grid = {
        'regressor__regressor__n_estimators': [150, 300, 500],
        'regressor__regressor__learning_rate': [0.05, 0.1, 0.2],
        'regressor__regressor__max_depth': [5, 7, 9], # Total: 18 combinaciones x 5 CV = 90 fits
        'regressor__regressor__min_child_weight':[1,3,5]
    }
    
    inicio_grid = time.time()
    grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='r2', n_jobs=1)
    grid_search.fit(X_train, y_train)
    tiempo_grid = time.time() - inicio_grid

    print(f"   Mejores Params (Grid): {grid_search.best_params_}")
    print(f"   Mejor R2 CV (Grid): {grid_search.best_score_:.4f}")
    print(f"   Tiempo de búsqueda: {tiempo_grid:.2f} segundos")

    # Estrategia 2: RandomizedSearch
    print("\n Estrategia 2: RandomizedSearchCV")
    param_dist = {
        'regressor__regressor__n_estimators': stats.randint(100, 1000),
        'regressor__regressor__learning_rate': stats.uniform(0.01, 0.3),
        'regressor__regressor__max_depth': stats.randint(3, 15),
        'regressor__regressor__min_child_weight':stats.randint(1,10),
        'regressor__regressor__subsample': stats.uniform(0.5, 0.5),
        'regressor__regressor__colsample_bytree': stats.uniform(0.5, 0.5)
    }
    
    inicio_random = time.time()
    random_search = RandomizedSearchCV(full_pipeline, param_distributions=param_dist, n_iter=50, cv=5, scoring='r2', n_jobs=1, random_state=42)
    random_search.fit(X_train, y_train)
    tiempo_random = time.time() - inicio_random

    print(f"   Mejores Params (Random): {random_search.best_params_}")
    print(f"   Mejor R2 CV (Random): {random_search.best_score_:.4f}")
    print(f"   Tiempo de búsqueda: {tiempo_random:.2f} segundos")

    # Ganador comparando los mejores R2 obtenidos en CV
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

    # Subimos a W&B
    wandb.log({
        "mejor_estrategia": mejor_estrategia,
        "mejor_n_estimators": mejores_params['regressor__regressor__n_estimators'],
        "mejor_learning_rate": mejores_params['regressor__regressor__learning_rate'],
        "mejor_max_depth": mejores_params['regressor__regressor__max_depth'],
        "mejor_min_child_weight": mejores_params.get("regressor__regressor__min_child_weight",1.0),
        "mejor_subsample": mejores_params.get('regressor__regressor__subsample', 1.0),
        "mejor_colsample_bytree": mejores_params.get('regressor__regressor__colsample_bytree', 1.0),
        "val_r2_cv": mejor_r2,
        "tiempo_grid_segundos": tiempo_grid,
        "tiempo_random_segundos": tiempo_random,
        "mercado": nombre_mercado,
        "modelo": "XGBoost (Tuning)"
    })
    
    run.finish()
    print("\n Búsqueda terminada. Copia los hiperparámetros ganadores para el script de evaluación.")

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_ventas_arboles.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_arboles.parquet")

    entrenar_xgboost_tuning(df_venta, "venta")
    entrenar_xgboost_tuning(df_alquiler, "alquiler")