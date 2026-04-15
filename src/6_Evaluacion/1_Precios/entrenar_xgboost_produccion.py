import pandas as pd
import numpy as np
import joblib
import wandb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

from utils.funciones_minio import crear_cliente_minio, bajar_minio

# Hiperparámetros ganadores de la fase de Tuning
MEJORES_PARAMS_XGB = {
    "venta": {
        "n_estimators": 895, 
        "learning_rate": 0.09869010575131419, 
        "max_depth": 7,
        "min_child_weight":6,
        "subsample": 0.9018360384495572,
        "colsample_bytree": 0.8121770240668966
    },
    "alquiler": {
        "n_estimators": 472, 
        "learning_rate": 0.039992474745400866, 
        "max_depth": 13,
        "min_child_weight": 8,
        "subsample": 0.8005575058716043,
        "colsample_bytree": 0.7229163764267956
    }
}

def entrenar_y_guardar_produccion(df, nombre_mercado):
    print(f"ENTRENANDO MODELO DE PRODUCCIÓN: {nombre_mercado.upper()}")
    print(f"Usando el 100% de los datos: {len(df)} registros.")

    # 1. Separación de variables
    X = df.drop(columns=['Precio'])
    
    if nombre_mercado == 'venta':
        print("Aplicando estrategia: Precio / m²")
        y = df['Precio'] / X['Superficie']
    else:
        print("Aplicando estrategia: Precio Total Absoluto")
        y = df['Precio']

    cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X[cat_cols] = X[cat_cols].fillna('Desconocido').astype(str)

    # 2. Preprocesamiento
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols),('cat', cat_transformer, cat_cols)])

    # 3. Ensamblaje del Modelo Final
    params = MEJORES_PARAMS_XGB[nombre_mercado]
    modelo_produccion = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', TransformedTargetRegressor(
            regressor=XGBRegressor(
                objective="reg:absoluteerror",
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                min_child_weight=params.get("min_child_weight", 1),
                subsample=params.get("subsample", 1.0),
                colsample_bytree=params.get("colsample_bytree", 1.0),
                random_state=42,
                n_jobs=-1
            ),
            func=np.log1p, inverse_func=np.expm1
        ))
    ])

    # 4. ENTRENAMIENTO (100% de los datos)
    print("Entrenando el modelo definitivo... (Esto puede tardar un poco)")
    modelo_produccion.fit(X, y)

    # 5. Guardado local
    nombre_archivo = f"modelo_produccion_{nombre_mercado}.pkl"
    joblib.dump(modelo_produccion, nombre_archivo)
    print(f"Modelo local guardado en disco como: {nombre_archivo}")

    # 6. MLOps: Registro en Weights & Biases
    print("Subiendo el modelo al Model Registry de W&B...")
    run = wandb.init(
        entity="pd1-c2526-team2", 
        project=f"modelo-precio-viviendas-{nombre_mercado}",
        name=f"produccion-final-{nombre_mercado}",
        job_type="model-registry"
    )
    
    # Creamos el artefacto
    artefacto = wandb.Artifact(
        name=f"xgboost-hibrido-{nombre_mercado}", 
        type='model',
        description=f"Modelo de producción para {nombre_mercado} entrenado con el 100% de los datos."
    )
    
    # Adjuntamos el archivo .pkl que acabamos de crear y lo subimos
    artefacto.add_file(nombre_archivo)
    run.log_artifact(artefacto)
    run.finish()
    print(f"¡Artefacto subido con éxito a W&B!")

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    
    # Descargamos los datos de minio
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_ventas_arboles.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_arboles.parquet")

    # Ejecutamos para ambos mercados
    entrenar_y_guardar_produccion(df_venta, "venta")
    entrenar_y_guardar_produccion(df_alquiler, "alquiler")