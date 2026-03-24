from xml.parsers.expat import model

import numpy as np
import pandas as pd
import wandb

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from utils.funciones_minio import crear_cliente_minio, bajar_minio


def filtrar_columnas(X : pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas que pueden empeorar KNN como:
    - identificadores
    - texto libre
    - URLs
    - columnas de cardinalidad muy alta sin valor geométrico útil
    """

    cols_to_drop = ["id","Nombre","Calle","Descripcion","Url"]
    cols_to_drop = [col for col in cols_to_drop if col in X.columns]

    #quitar Anuncia si tiene mas de 30 ya que puede perjurdicar al modelo
    if "Anuncia" in X.columns:
        n_unique = X["Anuncia"].nunique(dropna=True)
        if n_unique > 30:
            X = X.drop(columns=["Anuncia"])

    return X

def preparar_columnas(X: pd.DataFrame):
    """
    Detecta columnas numéricas y categóricas.
    """

    num_cols = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.to_list()
    num_fils = X.select_dtypes(include=["object","string","category","bool"]).columns.to_list()

    return num_cols,num_fils

def construir_preprocesador(num_cols, cat_cols):
    """
    Preprocesado para KNN:
    - numéricas: imputación mediana + RobustScaler
    - categóricas: imputación + OneHot con agrupación de categorías raras
    """

    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", RobustScaler())])
    
    #min_frequency = 0.01 para agrupar categorias muy raras
    cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",min_frequency=0.01,sparse_output=False))])

    preprocessor = ColumnTransformer(transformers=[("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)],remainder="drop")

    return preprocessor

def construir_modelos(num_cols,cat_cols):
    """
    Devuelve dos versiones del pipeline:
    - con selección de variables
    - sin selección de variables
    """

    modelos = []

    for freq in [0.005,0.01]:
        preprocessor = construir_preprocesador(num_cols, cat_cols)

        #pipeline sin selector
        pipeline_sin_selector = Pipeline(steps=[("preprocessor", preprocessor),("regressor", KNeighborsRegressor())])
        modelo_sin_selector =  TransformedTargetRegressor(regressor=pipeline_sin_selector,func=np.log1p,inverse_func=np.expm1)
        grid_sin_selector = [{"regressor__regressor__n_neighbors": [3, 5, 8, 12, 20, 30],
        "regressor__regressor__weights": ["distance"],"regressor__regressor__metric": ["manhattan", "euclidean"],
        "regressor__regressor__algorithm": ["auto", "brute"]},{"regressor__regressor__n_neighbors": [5, 8, 12, 20],
        "regressor__regressor__weights": ["uniform"],"regressor__regressor__metric": ["manhattan", "euclidean"],
        "regressor__regressor__algorithm": ["auto", "brute"]}]
        
        modelos.append({"nombre": f"knn_sin_selector_minfreq_{freq}","modelo": modelo_sin_selector,
        "param_grid":grid_sin_selector,"min_freq": freq,"usa_selector": False})

        #pipeline con selector
        pipeline_con_selector = Pipeline(steps=[("preprocessor", preprocessor),("selector",SelectKBest(score_func=mutual_info_regression, k=40)),("regressor", KNeighborsRegressor())])
        modelo_con_selector = TransformedTargetRegressor(regressor=pipeline_sin_selector,func=np.log1p,inverse_func=np.expm1)
        grid_con_selector =  [{"regressor__selector__k": [15, 30, 50, 80, "all"],
        "regressor__regressor__n_neighbors": [3, 5, 8, 12, 20, 30],"regressor__regressor__weights": ["distance"],
        "regressor__regressor__metric": ["manhattan", "euclidean"],"regressor__regressor__algorithm": ["auto", "brute"]},
        {"regressor__selector__k": [15, 30, 50, 80, "all"],"regressor__regressor__n_neighbors": [5, 8, 12, 20],
         "regressor__regressor__weights": ["uniform"],"regressor__regressor__metric": ["manhattan", "euclidean"],
         "regressor__regressor__algorithm": ["auto", "brute"]}]
        
        modelos.append({"nombre": f"knn_con_selector_minfreq_{freq}","modelo": modelo_con_selector,
        "param_grid":grid_con_selector,"min_freq": freq,"usa_selector": True})
    
    return modelos

def evaluar_modelo(best_model,X_test,y_test):
    """
    Evalúa el mejor modelo en test.
    """

    y_pred = best_model.predict(X_test)
    y_pred = np.maximum(y_pred,0)

    metricas = {"mae": mean_absolute_error(y_test,y_pred),
    "rmse": root_mean_squared_error(y_test,y_pred),"r2": r2_score(y_test,y_pred)}

    return metricas, y_pred

def entrenar_knn(df, nombre_mercado):
    """
    Entrena un modelo Knn y registra los resultados en Weights & Biases.
    """

    if "Precio" not in df.columns:
        raise ValueError("El dataframe no contiene la columna Precio")
    
    # 1.Separación de variables
    X = df.drop(columns=['Precio']).copy()
    X = filtrar_columnas(X)
    y = df['Precio'].copy()

    # 2.Filtrado de columnas
    X = filtrar_columnas(X)

    # 3.Limpieza de target
    mask_valid = y.notna() & np.isfinite(y) & (y > 0)
    X = X.loc[mask_valid].copy()
    y = y.loc[mask_valid].copy()

    if "Distrito" not in df.columns:
        raise ValueError("El dataframe no contiene la columna Distrito")
    
    # 4.Tipos
    num_cols, cat_cols = preparar_columnas(X)
    for col in cat_cols:
        X[col] = X[col].astype("string")
    
    # 5.Partición Estratificada (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=X['Distrito'])

    # 6.Elegir los modelos candidatos
    modelos = construir_modelos(num_cols,cat_cols)

    # 7.Cross-validation
    cross_validation = KFold(n_splits=5,shuffle=True,random_state=42)
    mejor_resultado = None
    mejor_modelo = None
    mejor_grid = None
    mejor_nombre = None

    # 8.Inicializamos W&B
    run = wandb.init(
        entity="pd1-c2526-team2",
        project=f"modelo-precio-viviendas-{nombre_mercado}",
        name=f"knn-{nombre_mercado}",
        job_type="model-training",
        group=nombre_mercado,
        config={
            "model": "KNNRegressor",
            "target_transform": "log1p",
            "scaler" : "RobustScaler",
            "feature_selection": "SelectKBest(mutual_info_regression)",
            "cross_validation": 5,
            "test_size" : 0.20,
            "random_state": 42,
        }
    )
    print(f"Buscando mejor Knn para {nombre_mercado}...")

    for candidato in modelos:
        #elegimos por MAE mas negativo para que este mas alineado con la regresion de precios
        grid_search = GridSearchCV(estimator=candidato["model"],param_grid=candidato["param_grid"],scoring="neg_mean_absolute_error",
        cv=cross_validation,n_jobs=-1,verbose=1,refit=True)
        grid_search.fit(X_train,y_train)

        best_model = grid_search.best_estimator_
        metricas_test, _ = evaluar_modelo(best_model,X_test,y_test)
        best_cross_validation_mae = grid_search.best_score_

        resumen = {"pipeline": candidato["nombre"],"usa_selector": candidato["usa_selector"],
        "min_frequency": candidato["min_frequency"],"cv_mae": best_cross_validation_mae,
        "test_mae": metricas_test["mae"],"test_rmse": metricas_test["rmse"],
        "test_r2": metricas_test["r2"],"best_params": grid_search.best_params_}

        print("Resultado candidato:")
        print(f"  CV MAE:  {best_cross_validation_mae:.4f}")
        print(f"  Test MAE:{metricas_test['mae']:.4f}")
        print(f"  Test RMSE:{metricas_test['rmse']:.4f}")
        print(f"  Test R2: {metricas_test['r2']:.4f}")
        print(f"  Best params: {grid_search.best_params_}")

        wandb.log({f"{candidato['nombre']}_cv_mae": best_cross_validation_mae,f"{candidato['nombre']}_test_mae": metricas_test["mae"],
        f"{candidato['nombre']}_test_rmse": metricas_test["rmse"],f"{candidato['nombre']}_test_r2": metricas_test["r2"]})

        if (mejor_resultado is None) or (metricas_test["mae"] < mejor_resultado["mae"]):
            mejor_resultado = {"mae": metricas_test["mae"],"rmse": metricas_test["rmse"],"r2": metricas_test["r2"],"best_score_cv_mae": 
            best_cross_validation_mae,"best_params": grid_search.best_params_,"best_pipeline": candidato["nombre"],
            "min_frequency": candidato["min_frequency"],"usa_selector": candidato["usa_selector"]}
            mejor_modelo = best_model
            mejor_grid = grid_search
            mejor_nombre = candidato["nombre"]

    print("\n=== MEJOR MODELO ===")
    print(f"Mercado: {nombre_mercado}")
    print(f"Pipeline: {mejor_nombre}")
    print(f"MAE:  {mejor_resultado['mae']:.4f}")
    print(f"RMSE: {mejor_resultado['rmse']:.4f}")
    print(f"R2:   {mejor_resultado['r2']:.4f}")
    print(f"CV MAE: {mejor_resultado['best_score_cv_mae']:.4f}")
    print("Mejores Parametros:")
    for k, v in mejor_resultado["best_params"].items():
        print(f"  - {k}: {v}")

    # Subimos a W&B
    wandb.log({"best_pipeline": mejor_resultado["best_pipeline"],"best_mae": mejor_resultado["mae"],
    "best_rmse": mejor_resultado["rmse"],"best_r2": mejor_resultado["r2"],
    "best_cv_mae": mejor_resultado["best_score_cv_mae"],"best_params": str(mejor_resultado["best_params"]),
    "best_min_frequency": mejor_resultado["min_frequency"],"best_usa_selector": mejor_resultado["usa_selector"]})
    run.finish()

    return best_model,mejor_resultado,mejor_grid

if __name__ == "__main__":

    cliente = crear_cliente_minio()
    
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_ventas_regresion.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_regresion.parquet")

    entrenar_knn(df_venta, "venta")
    entrenar_knn(df_alquiler, "alquiler")