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

def preparar_columnas(X: pd.DataFrame):

    num_cols = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.to_list()
    num_fils = X.select_dtypes(include=["object","string","category","bool"]).columns.to_list()

    return num_cols,num_fils


def construir_preprocesador(num_cols, cat_cols):

    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", RobustScaler())])
    
    #min_frequency = 0.01 para agrupar categorias muy raras
    cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",min_frequency=0.01,sparse_output=False))])

    preprocessor = ColumnTransformer(transformers=[("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)],remainder="drop")

    return preprocessor

def entrenar_knn(df, nombre_mercado):

    if "Precio" not in df.columns:
        raise ValueError("El dataframe no contiene la columna Precio")
    
    # 1.Separación de variables
    X = df.drop(columns=['Precio'])
    y = df['Precio'].copy()

    #limpieza de target
    mask_valid = y.notna() & np.isfinite(y) & (y > 0)
    X = X.loc[mask_valid].copy()
    y = y.loc[mask_valid].copy()

    if "Distrito" not in df.columns:
        raise ValueError("El dataframe no contiene la columna Distrito")
    
    num_cols, cat_cols = preparar_columnas(X)
    for col in cat_cols:
        X[col] = X[col].astype("string")
    
    # 2.Partición Estratificada (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=X['Distrito'])

    # 3.Preprocesado para Knn
    preprocessor = construir_preprocesador(num_cols, cat_cols)

    # 4.Pipeline Knn
    knn_pipeline = Pipeline(steps=[("preprocessor", preprocessor),("selector", 
    SelectKBest(score_func=mutual_info_regression, k=40)),("regressor", KNeighborsRegressor())])
    #se usa SelectKBest para reducir la dimensionalidad 

    #transformamos el target a logaritmico para que los precios se estabilizen
    model = TransformedTargetRegressor(regressor=knn_pipeline,func=np.log1p,inverse_func=np.expm1)

    # 5. Grid de parametros
    param_grid = [{"regressor__selector__k": [20, 40, 60, "all"],
                   "regressor__regressor__n_neighbors": [3, 5, 8, 12, 20, 30],
                   "regressor__regressor__weights": ["distance"],
                   "regressor__regressor__metric": ["manhattan", "euclidean"],
                   "regressor__regressor__algorithm": ["auto", "brute"]},
                  {"regressor__selector__k": [20, 40, 60, "all"],
                   "regressor__regressor__n_neighbors": [5, 8, 12, 20],
                   "regressor__regressor__weights": ["uniform"],
                   "regressor__regressor__metric": ["manhattan", "euclidean"],
                   "regressor__regressor__algorithm": ["auto", "brute"]}]
    
    cv = KFold(n_splits=5,shuffle=True,random_state=42)

    # Inicializamos W&B
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
            "cv": 5,
            "test_size" : 20,
            "random_state": 42,
            "param_grid": str(param_grid)
        }
    )
    print(f"Buscando mejor Knn para {nombre_mercado}...")

    #elegimos por MAE mas negativo para que este mas alineado con la regresion de precios
    grid_search = GridSearchCV(estimator=model,param_grid=param_grid,scoring="neg_mean_absolute_error",
    cv=cv,n_jobs=-1,verbose=1,refit=True)
    grid_search.fit(X_train,y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculamos las métricas finales
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "best_score_cv_mae": grid_search.best_score_,
        "best_params": grid_search.best_params_
    }

    print("\n=== RESULTADOS ===")
    print(f"Mercado: {nombre_mercado}")
    print(f"MAE:  {metrics[0]:.4f}")
    print(f"RMSE: {metrics[1]:.4f}")
    print(f"R2:   {metrics[2]:.4f}")
    print("Best params:")
    for k, v in grid_search.best_params_.items():
        print(f"  - {k}: {v}")

    # Subimos a W&B
    wandb.log(metrics)
    run.finish()

    return best_model,metrics

if __name__ == "__main__":

    cliente = crear_cliente_minio()
    
    df_venta = bajar_minio(cliente, "dataset_ml/precios/ventas", "df_ventas_regresion.parquet")
    df_alquiler = bajar_minio(cliente, "dataset_ml/precios/alquiler", "df_alquiler_regresion.parquet")

    entrenar_knn(df_venta, "venta")
    entrenar_knn(df_alquiler, "alquiler")



