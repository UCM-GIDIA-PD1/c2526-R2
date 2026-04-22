import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

from utils.funciones_minio import crear_cliente_minio, bajar_minio

# Hyperparameters based on previous tuning sessions
MEJORES_PARAMS_VENTA = {
    'n_estimators': 850,
    'learning_rate': 0.015,
    'max_depth': 8,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

MEJORES_PARAMS_ALQUILER = {
    'n_estimators': 750,
    'learning_rate': 0.02,
    'max_depth': 7,
    'min_child_weight': 4,
    'subsample': 0.85,
    'colsample_bytree': 0.85
}

def entrenar_y_guardar(df, nombre_mercado, params):
    print(f"\nENTRENANDO MODELO DE PRODUCCIÓN: {nombre_mercado.upper()}")

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

    # 2. Preprocesamiento
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

    # 3. Pipeline Base con XGBoost
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', TransformedTargetRegressor(
            regressor=XGBRegressor(
                objective="reg:absoluteerror", 
                random_state=42, 
                n_jobs=-1,
                **params
            ),
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])

    print("   Entrenando en el dataset completo...")
    full_pipeline.fit(X, y)
    print("   Modelo entrenado.")

    # 4. Guardar modelo completo (pipeline)
    # Ensure artifacts directory exists
    ROOT_DIR = Path(__file__).resolve().parents[3]
    artifacts_dir = ROOT_DIR / "src" / "model_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = artifacts_dir / f"{nombre_mercado}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(full_pipeline, f)
    print(f"   Modelo guardado en: {model_path}")

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    
    print("Descargando datasets desde Minio...")
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_ventas_arboles.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_arboles.parquet")

    entrenar_y_guardar(df_venta, "venta", MEJORES_PARAMS_VENTA)
    entrenar_y_guardar(df_alquiler, "alquiler", MEJORES_PARAMS_ALQUILER)
