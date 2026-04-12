import pandas as pd
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from utils.funciones_minio import crear_cliente_minio, bajar_minio

# Mejores parametros según la fase de tuning (copiados manualmente desde los resultados de W&B)

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

def evaluar_xgb_final_hibrido(df, nombre_mercado):
    print(f" EVALUACIÓN FINAL XGBOOST: {nombre_mercado.upper()}")

    # 1. Separación de variables (Para venta es mejor predecir precio/m2 y luego reconvertir, para alquiler predecimos precio total)
    X = df.drop(columns=['Precio'])
    
    if nombre_mercado == 'venta':
        print("Estrategia objetivo: Precio / m²")
        y = df['Precio'] / X['Superficie']
    else:
        print("Estrategia objetivo: Precio Total Absoluto")
        y = df['Precio']

    cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X[cat_cols] = X[cat_cols].fillna('Desconocido').astype(str)

    # 2. Partición Estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=X['Distrito']
    )

    # 3. Preprocesamiento
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols),('cat', cat_transformer, cat_cols)])

    # 4. Extraemos los mejores parámetros y montamos el Pipeline final
    params = MEJORES_PARAMS_XGB[nombre_mercado]
    modelo_final = Pipeline(steps=[
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

    print("Entrenando modelo definitivo...")
    modelo_final.fit(X_train, y_train)

    # 5. Predicciones
    y_pred_crudo = modelo_final.predict(X_test)

    if nombre_mercado == 'venta':
        # Reconvertimos multiplicando por la superficie
        y_pred_final = y_pred_crudo * X_test['Superficie'].values
        y_test_final = y_test * X_test['Superficie'].values
    else:
        y_pred_final = y_pred_crudo
        y_test_final = y_test

    # 6. Métricas globales
    mae = mean_absolute_error(y_test_final, y_pred_final)
    rmse = root_mean_squared_error(y_test_final, y_pred_final)
    r2 = r2_score(y_test_final, y_pred_final)

    print("\nMÉTRICAS GLOBALES EN TEST (EUROS TOTALES):")
    print(f"   · MAE:  {mae:,.2f} €")
    print(f"   · RMSE: {rmse:,.2f} €")
    print(f"   · R2:   {r2:.4f}")

    # 7. Métricas por distrito
    X_test_eval = X_test.copy()
    X_test_eval['Precio_Real'] = y_test_final
    X_test_eval['Precio_Predicho'] = y_pred_final
    X_test_eval['Error_Absoluto'] = abs(X_test_eval['Precio_Real'] - X_test_eval['Precio_Predicho'])
    
    error_por_distrito = X_test_eval.groupby('Distrito')['Error_Absoluto'].mean().sort_values(ascending=False)

    print("\nMAE POR DISTRITOS (Peores predicciones):")
    print(error_por_distrito.head(5).apply(lambda x: f"   {x:,.2f} €"))

    print("\nMAE POR DISTRITOS (Mejores predicciones):")
    print(error_por_distrito.tail(5).sort_values().apply(lambda x: f"   {x:,.2f} €"))

    # 8. Importancia de las variables
    importancias = modelo_final.named_steps['regressor'].regressor_.feature_importances_
    nombres_cat = modelo_final.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
    nombres_todas = num_cols + list(nombres_cat)
    
    df_importancia = pd.DataFrame({'Variable': nombres_todas, 'Importancia': importancias})
    top_importantes = df_importancia.sort_values(by='Importancia', ascending=False).head(5)
    
    print("\nTOP 5 VARIABLES MÁS IMPORTANTES:")
    for idx, row in top_importantes.iterrows():
        print(f"   {row['Variable']}: {row['Importancia']*100:.2f}%")

    # 9. Registro en W&B
    run = wandb.init(
        entity="pd1-c2526-team2",
        project=f"modelo-precio-viviendas-{nombre_mercado}",
        name=f"evaluacion-final-xgb-{nombre_mercado}",
        job_type="model-evaluation", 
        group=nombre_mercado
    )
    
    wandb.log({
        "estrategia_target": "Precio/m2" if nombre_mercado == 'venta' else "Precio Total",
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
        "n_estimators_usado": params["n_estimators"],
        "mercado": nombre_mercado,
        "modelo": "XGBoost (Híbrido Final)"
    })
    run.finish()

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_ventas_arboles.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_arboles.parquet")

    evaluar_xgb_final_hibrido(df_venta, "venta")
    evaluar_xgb_final_hibrido(df_alquiler, "alquiler")