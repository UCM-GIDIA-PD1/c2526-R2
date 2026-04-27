import pandas as pd
import numpy as np
import wandb
import time   
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
import optuna

from utils.funciones_minio import crear_cliente_minio, bajar_minio

def entrenar_xgboost_tuning(df, nombre_mercado):
    print(f"\nBÚSQUEDA DE HIPERPARÁMETROS (OPTUNA): {nombre_mercado.upper()}")

    # 1. Separación de variables
    X = df.drop(columns=['Precio'])
    
    if nombre_mercado == 'venta':
        print("   -> Target: Precio / m²")
        y = df['Precio'] / X['Superficie']
    else:
        print("   -> Target: Precio Total")
        y = df['Precio']

    cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X[cat_cols] = X[cat_cols].fillna('Desconocido').astype(str)

    # 2. Partición Estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=X['Distrito']
    )

    # 3. Preprocesamiento
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # 4. Pipeline Base con XGBoost
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', TransformedTargetRegressor(
            regressor=XGBRegressor(objective="reg:absoluteerror", random_state=42, n_jobs=-1),
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])

    run = wandb.init(
        entity="pd1-c2526-team2",
        project=f"modelo-precio-viviendas-{nombre_mercado}",
        name=f"xgb-tuning-optuna-{nombre_mercado}",
        job_type="hyperparameter-tuning",
        group=nombre_mercado
    )

    # 5. Definir la función objetivo de Optuna
    def objective(trial):
        params = {
            'regressor__regressor__n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'regressor__regressor__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'regressor__regressor__max_depth': trial.suggest_int('max_depth', 3, 15),
            'regressor__regressor__min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'regressor__regressor__subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'regressor__regressor__colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }
        
        modelo = clone(full_pipeline)
        modelo.set_params(**params)
        
        scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
        
        mape_positivo = -scores.mean() * 100
        return mape_positivo

    print("\n   Estrategia: Optuna (TPE Optimization) minimizando MAPE...")
    
    inicio_optuna = time.time()
    
    study = optuna.create_study(direction="minimize", study_name=f"XGB_{nombre_mercado}")
    study.optimize(objective, n_trials=50) 
    
    tiempo_optuna = time.time() - inicio_optuna

    mejores_params = study.best_params
    mejor_mape = study.best_value

    print(f"\n   Mejores Params (Optuna): {mejores_params}")
    print(f"   Mejor MAPE CV (Optuna): {mejor_mape:.2f} %")
    print(f"   Tiempo de búsqueda: {tiempo_optuna:.2f} segundos")

    # 6. Subimos a W&B
    wandb.log({
        "mejor_estrategia": "Optuna TPE",
        "mejor_n_estimators": mejores_params.get('n_estimators'),
        "mejor_learning_rate": mejores_params.get('learning_rate'),
        "mejor_max_depth": mejores_params.get('max_depth'),
        "mejor_min_child_weight": mejores_params.get('min_child_weight', 1.0),
        "mejor_subsample": mejores_params.get('subsample', 1.0),
        "mejor_colsample_bytree": mejores_params.get('colsample_bytree', 1.0),
        "val_mape_cv": mejor_mape,
        "tiempo_optuna_segundos": tiempo_optuna,
        "mercado": nombre_mercado,
        "modelo": "XGBoost Híbrido (Optuna Tuning)"
    })
    
    run.finish()

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_venta_xgboost.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_xgboost.parquet")

    entrenar_xgboost_tuning(df_venta, "venta")
    entrenar_xgboost_tuning(df_alquiler, "alquiler")