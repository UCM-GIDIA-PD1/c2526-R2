import numpy as np
import pandas as pd
import wandb

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler,StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

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
    X = X.drop(columns=cols_to_drop, errors="ignore")

    #quitar Anuncia si tiene mas de 30 ya que puede perjurdicar al modelo
    if "Anuncia" in X.columns:
        n_unique = X["Anuncia"].nunique(dropna=True)
        if n_unique > 30:
            X = X.drop(columns=["Anuncia"],errors="ignore")

    return X

def preparar_columnas(X: pd.DataFrame):
    """
    Detecta columnas numéricas y categóricas.
    """

    num_cols = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.to_list()
    num_fils = X.select_dtypes(include=["object","string","category","bool"]).columns.to_list()

    return num_cols,num_fils

def construir_preprocesador(num_cols, cat_cols,scaler,min_frequency):
    """
    Preprocesado para KNN:
    - numéricas: imputación mediana + RobustScaler
    - categóricas: imputación + OneHot con agrupación de categorías raras
    """

    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", scaler)])
    
    #min_frequency = 0.01 para agrupar categorias muy raras
    cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",min_frequency=min_frequency,sparse_output=False))])

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
    scalers = {"robust":RobustScaler(),"standard":StandardScaler(),"minmax":MinMaxScaler()}
    min_frequencies = [0.005,0.01,0.02]

    for scaler_name, scaler in scalers.items():
        for freq in min_frequencies:
            preprocessor = construir_preprocesador(num_cols=num_cols, cat_cols=cat_cols,scaler=scaler,min_frequency=freq)

            #pipeline sin selector
            pipeline_sin_selector = Pipeline(steps=[("preprocessor", preprocessor),("pca",PCA()),("regressor", KNeighborsRegressor())])
            modelo_sin_selector =  TransformedTargetRegressor(regressor=pipeline_sin_selector,func=np.log1p,inverse_func=np.expm1)
            grid_sin_selector = [{"regressor__pca__n_components":[0.80,0.90,0.95],"regressor__regressor__n_neighbors": [3, 5, 10, 20, 40,80],
            "regressor__regressor__weights": ["distance"],"regressor__regressor__p":[1,2],
            "regressor__regressor__algorithm": ["auto", "brute"],"regressor__regressor__leaf_size":[20,30,50]},{"regressor__pca_n_components":[0.80,0.90,0.95],"regressor__regressor__n_neighbors": [5, 10, 20, 40,80],
            "regressor__regressor__weights": ["uniform"],"regressor__regressor__p":[1,2],"regressor__regressor__algorithm": ["auto", "brute"],"regressor__regressor__leaf_size":[20,30,50]}]
        
            modelos.append({"nombre": f"knn_sin_selector_scaler_{scaler_name}_minfreq_{freq}","modelo": modelo_sin_selector,
            "param_grid":grid_sin_selector,"min_frequency": freq,"usa_selector": False,"scaler":scaler})

            #pipeline con selector
            pipeline_con_selector = Pipeline(steps=[("preprocessor", preprocessor),("selector",SelectKBest(score_func=mutual_info_regression)),("pca",PCA()),("regressor", KNeighborsRegressor())])
            modelo_con_selector = TransformedTargetRegressor(regressor=pipeline_con_selector,func=np.log1p,inverse_func=np.expm1)
            grid_con_selector =  [{"regressor__selector__k": [20, 50, 100, 200, "all"],"regressor__pca_n_components":[0.80,0.90,0.95],
            "regressor__regressor__n_neighbors": [3, 5, 10, 20, 40, 80],"regressor__regressor__weights": ["distance"], "regressor__regressor__p":[1,2],
            "regressor__regressor__algorithm": ["auto", "brute"],"regressor__regressor__leaf_size":[20,30,50]},
            {"regressor__selector__k": [20, 50, 100, 200,"all"],"regressor__pca_n_components":[0.80,0.90,0.95],"regressor__regressor__n_neighbors": [5, 10, 20, 40,80],
            "regressor__regressor__weights": ["uniform"],"regressor__regressor__p":[1,2],"regressor__regressor__algorithm": ["auto", "brute"],"regressor__regressor_leaf_size":[20,30,50]}]
        
            modelos.append({"nombre": f"knn_con_selector_scaler_{scaler_name}_minfreq_{freq}","modelo": modelo_con_selector,
            "param_grid":grid_con_selector,"min_frequency": freq,"usa_selector": True,"scaler":scaler})

    return modelos

def evaluar_modelo(best_model,X_test,y_test):
    """
    Evalúa el mejor modelo en test.
    """

    y_pred = best_model.predict(X_test)
    y_pred = np.maximum(y_pred,0)

    metricas = {"test_mae": mean_absolute_error(y_test,y_pred),
    "test_rmse": np.sqrt(mean_squared_error(y_test,y_pred)),"test_r2": r2_score(y_test,y_pred)}

    return metricas, y_pred

def entrenar_knn(df, nombre_mercado):
    """
    Entrena un modelo Knn y registra los resultados en Weights & Biases.
    """

    if "Precio" not in df.columns:
        raise ValueError("El dataframe no contiene la columna Precio")
    
    if "Distrito" not in df.columns:
        raise ValueError("El dataframe no contiene la columna Distrito")
    
    # 1.Separación de variables
    X = df.drop(columns=['Precio']).copy()
    y = df['Precio'].copy()

    # 2.Filtrado de columnas
    X = filtrar_columnas(X)

    # 3.Limpieza de target
    mask_valid = y.notna() & np.isfinite(y) & (y > 0)
    X = X.loc[mask_valid].copy()
    y = y.loc[mask_valid].copy()

    # 4.Tipos
    num_cols, cat_cols = preparar_columnas(X)
    for col in cat_cols:
        X[col] = X[col].replace({pd.NA: np.nan}).astype("object")
    
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
            "modelo": "KNNRegressor",
            "target_transform": "log1p",
            "cross_validation": 5,
            "test_size" : 0.20,
            "random_state": 42,
            "mejoras":["PCA","comparacion_scalers","onehot_min_frequency_variable","grid_knn_ampliado","opcion_con_y_sin_selector"]
        }
    )
    print(f"Buscando mejor Knn para {nombre_mercado}...")

    for candidato in modelos:
        print(f"\nEntrenando candidato: {candidato['nombre']}")

        #elegimos por MAE mas negativo para que este mas alineado con la regresion de precios
        grid_search = GridSearchCV(estimator=candidato["modelo"],param_grid=candidato["param_grid"],scoring="neg_mean_absolute_error",
        cv=cross_validation,n_jobs=-1,verbose=1,refit=True)
        grid_search.fit(X_train,y_train)

        best_model = grid_search.best_estimator_
        metricas_test, _ = evaluar_modelo(best_model,X_test,y_test)
        cv_mae = -grid_search.best_score_

        resumen = {"pipeline": candidato["nombre"],"usa_selector": candidato["usa_selector"],
        "min_frequency": candidato["min_frequency"],"scaler":candidato["scaler"],"cv_mae": cv_mae,
        "test_mae": metricas_test["test_mae"],"test_rmse": metricas_test["test_rmse"],
        "test_r2": metricas_test["test_r2"],"best_params": grid_search.best_params_}

        print("Resultado candidato:")
        print(f"  CV MAE:  {cv_mae:.4f}")
        print(f"  Test MAE:{metricas_test['test_mae']:.4f}")
        print(f"  Test RMSE:{metricas_test['test_rmse']:.4f}")
        print(f"  Test R2: {metricas_test['test_r2']:.4f}")
        print(f"  Best params: {grid_search.best_params_}")

        wandb.log({f"{candidato['nombre']}_cv_mae": cv_mae,f"{candidato['nombre']}_test_mae": metricas_test["test_mae"],
        f"{candidato['nombre']}_test_rmse": metricas_test["test_rmse"],f"{candidato['nombre']}_test_r2": metricas_test["test_r2"]})

        if (mejor_resultado is None) or (metricas_test["test_mae"] < mejor_resultado["test_mae"]):
            mejor_resultado = {"test_mae": metricas_test["test_mae"],"test_rmse": metricas_test["test_rmse"],"test_r2": metricas_test["test_r2"],
            "cv_mae": cv_mae,"best_params": grid_search.best_params_,"best_pipeline": candidato["nombre"],
            "min_frequency": candidato["min_frequency"],"usa_selector": candidato["usa_selector"],"scaler":candidato["scaler"]}
            mejor_modelo = best_model
            mejor_grid = grid_search
            mejor_nombre = candidato["nombre"]

    print("\n=== MEJOR MODELO ===")
    print(f"Mercado: {nombre_mercado}")
    print(f"Pipeline: {mejor_nombre}")
    print(f"test_MAE:  {mejor_resultado['test_mae']:.4f}")
    print(f"test_RMSE: {mejor_resultado['test_rmse']:.4f}")
    print(f"test_R2:   {mejor_resultado['test_r2']:.4f}")
    print(f"CV_MAE: {mejor_resultado['cv_mae']:.4f}")
    print("Mejores Parametros:")
    for k, v in mejor_resultado["best_params"].items():
        print(f"  - {k}: {v}")

    # Subimos a W&B
    wandb.log({"best_pipeline": mejor_resultado["best_pipeline"],"test_mae": mejor_resultado["test_mae"],
    "test_rmse": mejor_resultado["test_rmse"],"test_r2": mejor_resultado["test_r2"],
    "cv_mae": mejor_resultado["cv_mae"],"best_params": str(mejor_resultado["best_params"]),
    "best_min_frequency": mejor_resultado["min_frequency"],"best_usa_selector": mejor_resultado["usa_selector"],"best_scaler":mejor_resultado["scaler"]})
    run.finish()

    return best_model,mejor_resultado,mejor_grid

if __name__ == "__main__":

    cliente = crear_cliente_minio()
    
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_venta_limpio.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_limpio.parquet")

    entrenar_knn(df_venta, "venta")
    entrenar_knn(df_alquiler, "alquiler")