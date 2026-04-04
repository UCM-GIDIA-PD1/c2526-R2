import pandas as pd
import numpy as np
import wandb

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from utils.funciones_minio import crear_cliente_minio, bajar_minio

# Mejores parametros según la fase de tuning (copiados manualmente desde los resultados de W&B)
MEJORES_PARAMS_KNN = {
    "venta": {
        "scaler": "robust", 
        "min_frequency": 0.005,
        "usa_selector": True,
        "selector_k": 80,
        "pca_n_components": 0.95,
        "n_neighbors": 20,
        "weights": 'distance',
        "p": 2,
        "algorithm": 'auto',
        "leaf_size": 30
    },
    "alquiler": {
        "scaler": "robust", 
        "min_frequency": 0.005,
        "usa_selector": True,
        "selector_k": 80,
        "pca_n_components": 0.95,
        "n_neighbors": 20,
        "weights": 'distance',
        "p": 2,
        "algorithm": 'auto',
        "leaf_size": 30
    }
}

def filtrar_columnas(X:pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas que pueden empeorar KNN:
    - identificadores
    - texto libre
    - URLs
    - columnas de cardinalidad muy alta sin valor geométrico útil
    """
    cols_to_drop = ["id", "Nombre", "Calle", "Descripcion", "Url"]
    cols_to_drop = [col for col in cols_to_drop if col in X.columns]
    X = X.drop(columns=cols_to_drop,errors='ignore')

    if "Anuncia" in X.columns:
        n_unique = X["Anuncia"].nunique(dropna=True)
        if n_unique > 30:
            X = X.drop(columns=["Anuncia"], errors='ignore')
    
    return X

def preparar_columnas(X:pd.DataFrame):
    """
    Detecta columnas numéricas y categóricas.
    """
    num_cols = X.select_dtypes(include=['int64', 'float64',"int32", "float32"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['int64', 'float64',"int32", "float32"]).columns.tolist()

    return num_cols, cat_cols

def obtener_scaler(nombre_scaler:str):
    """
    Devuelve el scaler correspondiente.
    """
    if nombre_scaler == "robust":
        return RobustScaler()
    elif nombre_scaler == "standard":
        return StandardScaler()
    elif nombre_scaler == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(f"Scaler desconocido: {nombre_scaler}")

def construir_preprocesador(num_cols, cat_cols, scaler,min_frequency):
    """
    Preprocesado para KNN:
    - numéricas: imputación mediana + escalado
    - categóricas: imputación + OneHot con agrupación de categorías raras
    """
    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)])
    cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")), ("onehot", OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency = min_frequency, sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)], remainder='drop')

    return preprocessor

def construir_modelo_final_knn(num_cols, cat_cols, params):
    """
    Recrea el pipeline final del mejor KNN encontrado en tuning.
    """
    scaler = obtener_scaler(params["scaler"])
    preprocessor = construir_preprocesador(num_cols=num_cols, cat_cols=cat_cols, scaler=scaler, min_frequency=params["min_frequency"])
    pasos = [("preprocessor", preprocessor)]

    if params["usa_selector"]:
        pasos.append(("selector", SelectKBest(score_func=mutual_info_regression, k=params["selector_k"])))
    
    pasos.append(("pca", PCA(n_components=params["pca_n_components"])))
    pasos.append(("regressor", KNeighborsRegressor(n_neighbors=params["n_neighbors"], weights=params["weights"], p=params["p"], algorithm=params["algorithm"], leaf_size=params["leaf_size"])))

    pipeline = Pipeline(steps=pasos)
    modelo_final = TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)

    return modelo_final

def evaluar_knn_final(df, nombre_mercado):
    """
    Evalúa el modelo KNN final con los mejores parámetros encontrados en tuning.
    """
    print(f" EVALUACIÓN FINAL DEL MODELO KNN: {nombre_mercado.upper()}")

    if "Precio" not in df.columns:
        raise ValueError("La columna Precio no está en el DataFrame")
    if "Distrito" not in df.columns:
        raise ValueError("La columna Distrito no está en el DataFrame")
    
    #1. Separacion de variables
    X = df.drop(columns=['Precio']).copy()
    y = df['Precio'].copy()

    #2. Filtrado de columnas problemáticas para KNN
    X = filtrar_columnas(X)

    #3. Limpieza del target
    mask_valid = y.notna() & np.isfinite(y) & (y > 0)
    X = X.loc[mask_valid].copy()
    y = y.loc[mask_valid].copy()

    #4. Detección de tipos y limpieza de columnas categóricas
    num_cols, cat_cols = preparar_columnas(X)
    for col in cat_cols:
        X[col] = X[col].replace({pd.NA: np.nan}).astype("object")
    
    #5. Partición estratificada
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=X['Distrito'])

    #6. Construcción del modelo con los mejores parámetros
    params = MEJORES_PARAMS_KNN[nombre_mercado]

    #7. Construcción del pipeline final
    modelo_final = construir_modelo_final_knn(num_cols, cat_cols, params)
    print(f"Entrenando modelo definitivo con parámetros: {params}...")
    modelo_final.fit(X_train, y_train)

    #8. Predicción sobre el Test
    y_pred = modelo_final.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    #9. Métricas globales
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n Métricas globales en test:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")

    #10. Métricas por distrito
    X_test_eval = X_test.copy()
    X_test_eval["Precio_Real"] = y_test.values
    X_test_eval["Precio_Predicho"] = y_pred
    X_test_eval["Error_Absoluto"] = np.abs(X_test_eval["Precio_Real"] - X_test_eval["Precio_Predicho"])
    error_por_distrito = X_test_eval.groupby("Distrito")["Error_Absoluto"].mean().sort_values(ascending=False)

    print("\n MAE por distrito (top 5 con mayor error - peores predicciones):")
    print(error_por_distrito.head(5).apply(lambda x: f"{x:.2f} €"))
    print("\n MAE por distrito (top 5 con menor error - mejores predicciones):")
    print(error_por_distrito.tail(5).sort_values().apply(lambda x: f"{x:.2f} €"))

    #11. Importancia de variables
    print("\n Importancia de variables (top 10):")
    sample_size = min(1000, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=42)
    y_test_sample = y_test.loc[X_test_sample.index]
    resultado_perm = permutation_importance(modelo_final, X_test_sample, y_test_sample,scoring='neg_mean_absolute_error', n_repeats=5, random_state=42, n_jobs=-1)

    df_importancia = pd.DataFrame({"Variable": X_test.columns, "Importancia": resultado_perm.importances_mean}).sort_values(by="Importancia", ascending=False)
    top_importantes = df_importancia.head(10)
    for _, row in top_importantes.iterrows():
        print(f"   {row['Variable']}: {row['Importancia']:.4f}")

    #12. Registro en W&B
    run = wandb.init(
        entity="pd1-c2526-team2",
        project=f"modelo-precio-viviendas-{nombre_mercado}", 
        name=f"evaluacion-final-knn-{nombre_mercado}",
        job_type="model-evaluation",
        group=nombre_mercado,
        config={
            "modelo": "KNN (Final)",
            "scaler": params["scaler"],
            "min_frequency": params["min_frequency"],
            "usa_selector": params["usa_selector"],
            "selector_k": params["selector_k"],
            "pca_n_components": params["pca_n_components"],
            "n_neighbors": params["n_neighbors"],
            "weights": params["weights"],
            "p": params["p"],
            "algorithm": params["algorithm"],
            "leaf_size": params["leaf_size"],
            "target_transform": "log1p"
        }
    )
    wandb.log({
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
        "mercado": nombre_mercado,
        "modelo": "KNN (Final)",
        "scaler_usado": params["scaler"],
        "min_frequency_usado": params["min_frequency"],
        "usa_selector": int(params["usa_selector"]),
        "selector_k_usado": params["selector_k"] if params["usa_selector"] else -1,
        "pca_n_components_usado": params["pca_n_components"],
        "n_neighbors_usado": params["n_neighbors"],
        "weights_usado": params["weights"],
        "p_usado": params["p"],
        "leaf_size_usado": params["leaf_size"]
    })
    run.finish()

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_venta_limpio.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_limpio.parquet")

    evaluar_knn_final(df_venta, "venta")
    evaluar_knn_final(df_alquiler, "alquiler")