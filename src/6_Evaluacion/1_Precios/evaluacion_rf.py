import pandas as pd
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error

from utils.funciones_minio import crear_cliente_minio, bajar_minio

# Mejores parametros según la fase de tuning (copiados manualmente desde los resultados de W&B)
MEJORES_PARAMS_RF = {
    "venta": {
        "n_estimators": 400, 
        "max_depth": 25, 
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0
    },
    "alquiler": {
        "n_estimators": 692, 
        "max_depth": None, 
        "min_samples_split": 4,
        "min_samples_leaf": 3,
        "max_features": 1.0
    }
}

def evaluar_rf_final(df, nombre_mercado):
    print(f" EVALUACIÓN FINAL DEL MODELO RANDOM FOREST: {nombre_mercado.upper()}")
    
    X = df.drop(columns=['Precio'])
    y = df['Precio']

    cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X[cat_cols] = X[cat_cols].fillna('Desconocido')
    X[cat_cols] = X[cat_cols].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=X['Distrito']
    )

    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols),('cat', cat_transformer, cat_cols)])

    # Extraemos los mejores parámetros para el mercado actual
    params = MEJORES_PARAMS_RF[nombre_mercado]
    
    modelo_final = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=42,
            n_jobs=-1
        ))
    ])

    print(f"Entrenando modelo definitivo con parámetros: {params}...")
    modelo_final.fit(X_train, y_train)

    # 5. Predicción sobre el Test
    y_pred = modelo_final.predict(X_test)

    # Métricas globales
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    # Métricas por distrito
    X_test_eval = X_test.copy()
    X_test_eval['Precio_Real'] = y_test
    X_test_eval['Precio_Predicho'] = y_pred
    X_test_eval['Error_Absoluto'] = abs(X_test_eval['Precio_Real'] - X_test_eval['Precio_Predicho'])
    X_test_eval['Error_Porcentual'] = (X_test_eval['Error_Absoluto'] / X_test_eval['Precio_Real']) * 100

    mape_por_distrito = X_test_eval.groupby('Distrito')['Error_Porcentual'].mean().sort_values(ascending=False)

    print("\n MAPE POR DISTRITOS (Top 5 con MAYOR error - Peores predicciones):")
    print(mape_por_distrito.head(5).apply(lambda x: f"   {x:.2f} %"))

    print("\n MAPE POR DISTRITOS (Top 5 con MENOR error - Mejores predicciones):")
    print(mape_por_distrito.tail(5).sort_values().apply(lambda x: f"   {x:.2f} %"))

    # Importancia de las variables (Específico de Random Forest)
    print("\n IMPORTANCIA DE LAS VARIABLES (Top 10):")
    nombres_cat = modelo_final.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
    nombres_todas = num_cols + list(nombres_cat)
    
    # Extraemos las feature_importances_ del Random Forest
    importancias = modelo_final.named_steps['regressor'].feature_importances_
    
    df_importancia = pd.DataFrame({'Variable': nombres_todas, 'Importancia': importancias})
    top_importantes = df_importancia.sort_values(by='Importancia', ascending=False).head(10)
    
    for idx, row in top_importantes.iterrows():
        # Lo multiplicamos por 100 para que se lea como un porcentaje de importancia
        print(f"   {row['Variable']}: {row['Importancia']*100:.2f}%")

    # Registro en W&B
    run = wandb.init(
        entity="pd1-c2526-team2",
        project=f"modelo-precio-viviendas-{nombre_mercado}",
        name=f"evaluacion-final-rf-{nombre_mercado}",
        job_type="model-evaluation",
        group=nombre_mercado
    )
    
    wandb.log({
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
        "test_mape": mape,
        "n_estimators_usado": params["n_estimators"],
        "max_depth_usado": params["max_depth"] if params["max_depth"] is not None else 0, # W&B no traga None
        "mercado": nombre_mercado,
        "modelo": "RandomForest (Final)"
    })
    run.finish()

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_ventas_arboles.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_arboles.parquet")

    evaluar_rf_final(df_venta, "venta")
    evaluar_rf_final(df_alquiler, "alquiler")