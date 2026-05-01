import pandas as pd
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error

from utils.funciones_minio import crear_cliente_minio, bajar_minio

# Mejores alphas según la fase de tuning (copiados manualmente desde los resultados de W&B)
MEJORES_ALPHAS = {
    "venta": 127.07,
    "alquiler": 1
}

def evaluar_modelo_final(df, nombre_mercado):
    """
    Evalúa el modelo Lasso final con el mejor alpha encontrado en la fase de tuning.
    """
    print("Evaluación del modelo final de Lasso para el mercado de", nombre_mercado)
    # 1. Separación de variables y Parche de Textos
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

    # 3. Recreamos el Pipeline idéntico
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])

    cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols),('cat', cat_transformer, cat_cols)])

    # 4. Instanciamos el modelo con el mejor alpha encontrado en la fase de tuning 
    mejor_alpha = MEJORES_ALPHAS[nombre_mercado]
    modelo_final = Pipeline(steps=[('preprocessor', preprocessor),('regressor', Lasso(alpha=mejor_alpha, max_iter=50000, random_state=42))])

    # Entrenamos con todos los datos de Train
    print(f"Entrenando modelo definitivo con Alpha = {mejor_alpha}...")
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

    # Calculamos la media de error por distrito y ordenamos de peor a mejor
    mape_por_distrito = X_test_eval.groupby('Distrito')['Error_Porcentual'].mean().sort_values(ascending=False)

    print("\nMAPE POR DISTRITOS (Peores predicciones en %):")
    print(mape_por_distrito.head(5).apply(lambda x: f"   {x:.2f} %"))

    print("\nMAPE POR DISTRITOS (Mejores predicciones en %):")
    print(mape_por_distrito.tail(5).sort_values().apply(lambda x: f"   {x:.2f} %"))

    # 8. Influencia de las variables (coeficientes del Lasso)
    print("\nTop 5 variables que más SUMAN al precio:")
    nombres_cat = modelo_final.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
    nombres_todas = num_cols + list(nombres_cat)
    coeficientes = modelo_final.named_steps['regressor'].coef_
    df_coef = pd.DataFrame({'Variable': nombres_todas, 'Impacto_Euros': coeficientes})
    
    top_positivas = df_coef.sort_values(by='Impacto_Euros', ascending=False).head(5)
    for idx, row in top_positivas.iterrows():
        print(f"   {row['Variable']}: +{row['Impacto_Euros']:,.2f} €")

    print("\nTop 5 variables que más RESTAN al precio:")
    top_negativas = df_coef.sort_values(by='Impacto_Euros', ascending=True).head(5)
    for idx, row in top_negativas.iterrows():
        print(f"   {row['Variable']}: {row['Impacto_Euros']:,.2f} €")

    # Registro en W&B
    run = wandb.init(
        entity="pd1-c2526-team2",
        project=f"modelo-precio-viviendas-{nombre_mercado}",
        name=f"evaluacion-final-lasso-{nombre_mercado}",
        job_type="model-evaluation",
        group=nombre_mercado
    )
    
    wandb.log({
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
        "test_mape": mape,
        "alpha_usado": mejor_alpha,
        "mercado": nombre_mercado,
        "modelo": "Lasso (Final)"
    })
    run.finish()

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_ventas_regresion.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_regresion.parquet")

    evaluar_modelo_final(df_venta, "venta")
    evaluar_modelo_final(df_alquiler, "alquiler")