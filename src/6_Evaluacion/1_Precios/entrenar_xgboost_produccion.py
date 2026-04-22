import pandas as pd
import numpy as np
import joblib
import wandb
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

from utils.funciones_minio import crear_cliente_minio, bajar_minio

# Hiperparámetros ganadores de la fase de Tuning
MEJORES_PARAMS_XGB = {
    "venta": {
        "n_estimators": 849, 
        "learning_rate": 0.043396975107577444, 
        "max_depth": 8,
        "min_child_weight":7,
        "subsample": 0.8281908057671924,
        "colsample_bytree": 0.8572025726581108
    },
    "alquiler": {
        "n_estimators": 915, 
        "learning_rate": 0.05175244756697071, 
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.6685717435581119,
        "colsample_bytree": 0.5846115448733098
    }
}

def entrenar_y_guardar_produccion(df, nombre_mercado):
    print(f"\nENTRENANDO MODELO DE PRODUCCIÓN: {nombre_mercado.upper()}")
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

    # 2. Partición 80/20 estratificada por 'Distrito' según la regla de negocio
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_percentage_error
    
    print("Realizando partición 80% entrenamiento y 20% test, estratificada por 'Distrito'...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=X['Distrito'])

    # 3. Preprocesamiento
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols),('cat', cat_transformer, cat_cols)])

    # 4. Ensamblaje del Modelo Final
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

    # 5. CROSS VALIDATION EN TRAINING (80%)
    print("Aplicando 5-Fold Cross Validation sobre conjunto de entrenamiento...")
    scores = cross_val_score(modelo_produccion, X_train, y_train, cv=5, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
    print(f"-> [CV 5-Folds] MAPE Promedio: {-scores.mean()*100:.2f}%")
    
    # 6. EVALUACIÓN EN HOLD-OUT (20%)
    print("Entrenando temporalmente para hold-out...")
    modelo_produccion.fit(X_train, y_train)
    y_pred_test = modelo_produccion.predict(X_test)
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    print(f"-> [Test] MAPE en Test dataset (20%): {test_mape:.2f}%\n")

    # 7. ENTRENAMIENTO DEFINTIVO (100% de los datos)
    print("Entrenando el modelo definitivo con el 100% de los datos para producción...")
    modelo_produccion.fit(X, y)
    
    # También asegurar la persistencia en local para FastAPI
    import pathlib
    local_artifacts_dir = pathlib.Path(__file__).resolve().parents[3] / "src" / "model_artifacts"
    local_artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo_produccion, local_artifacts_dir / f"{nombre_mercado}_model.pkl")

    # 5. Guardado temporal y subida a W&B
    nombre_archivo = f"modelo_produccion_{nombre_mercado}.pkl"
    
    try:
        # Guardamos en disco de forma temporal
        joblib.dump(modelo_produccion, nombre_archivo)
        
        # 6. Registro en Weights & Biases
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
        
        # Adjuntamos el archivo .pkl y lo subimos
        artefacto.add_file(nombre_archivo)
        run.log_artifact(artefacto)
        run.finish()
        print(f"¡Artefacto subido con éxito a W&B!")

    finally:
        # 7. Limpieza del archivo temporal
        if os.path.exists(nombre_archivo):
            os.remove(nombre_archivo)
            print(f"El archivo temporal '{nombre_archivo}' ha sido eliminado de tu equipo.")

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    
    # Descargamos los datos de minio
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_ventas_arboles.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_arboles.parquet")

    # Ejecutamos para ambos mercados
    entrenar_y_guardar_produccion(df_venta, "venta")
    entrenar_y_guardar_produccion(df_alquiler, "alquiler")