import pandas as pd
import numpy as np 
import joblib
import wandb
import os
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
from utils.funciones_minio import crear_cliente_minio,bajar_minio



def basline_cat_max(X_train,X_test,y_train,y_test):
    run = wandb.init(entity = "pd1-c2526-team2",project="clasificador-imagen",job_type = "train")

    modelo_baseline = DummyClassifier(strategy='most_frequent')

    modelo_baseline.fit(X_train, y_train)

    ruta_mdl = os.path.join(run.dir, "modelo_baseline.joblib")
    joblib.dump(modelo_baseline, ruta_mdl)

    predicciones = modelo_baseline.predict(X_test)
    predicciones_proba = modelo_baseline.predict_proba(X_test)

    wandb.sklearn.plot_classifier(
        modelo_baseline, X_train, X_test, y_train, y_test, 
        predicciones, predicciones_proba, 
        labels=['Cocina', 'Dormitorio', 'Salón', 'Banyo'], 
        model_name='Baseline'
    )
    precision_baseline = accuracy_score(y_test, predicciones)
    print(f" Precisión del Baseline: {precision_baseline * 100:.2f}%")

    run.finish()

def preparar_dataset(df:pd.DataFrame):
    X = np.stack(df['embedding'].values)
    y = df['clase'].values.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, stratify=y)
    return X_train,X_test,y_train,y_test

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    ruta_ml = "dataset_ml"
    fichero = "embeddings_imagenes.parquet"
    df = bajar_minio(cliente,ruta_ml,fichero)
    X_train,X_test,y_train,y_test = preparar_dataset(df)
    basline_cat_max(X_train,X_test,y_train,y_test)