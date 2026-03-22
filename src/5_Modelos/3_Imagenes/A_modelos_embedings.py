import pandas as pd
import numpy as np 
import joblib
import wandb
import os
import tempfile
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score,recall_score
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
from utils.funciones_minio import crear_cliente_minio,bajar_minio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

clases = ['Cocina', 'Dormitorio', 'Salón', 'Banyo']


def entrenar_arbol_decision(X_train, X_test, y_train, y_test, etiquetas_clases, max_profundidad=40):
    """
    Entrena un DecisionTreeClassifier, evalúa sus métricas y 
    sube tanto los resultados como el modelo físico a Weights & Biases.
    """
    
    run = wandb.init(
        entity = "pd1-c2526-team2",
        project="clasificador-imagenes", 
        name="busqueda-hiperparametros-arbol",
        job_type="hyperparameter-tuning"
    )

    wandb.define_metric("profundidad_arbol")
    
    wandb.define_metric("accuracy", step_metric="profundidad_arbol")
    wandb.define_metric("f1_score", step_metric="profundidad_arbol")
    wandb.define_metric("recall", step_metric="profundidad_arbol")
    
    rango_profundidades = np.arange(1, max_profundidad + 1, 2)
    
    for p in tqdm(rango_profundidades,desc = "Entrenando Arboles de Decision con varias profundidades"):
        modelo = DecisionTreeClassifier(max_depth=p, random_state=42, criterion="entropy")
        modelo.fit(X_train, y_train)
        
        y_pred = modelo.predict(X_test)
        y_pred_train = modelo.predict(X_train)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        train_acc = accuracy_score(y_train, y_pred_train)
        
        wandb.log({
            "profundidad_arbol": p,  
            "accuracy": acc,        
            "f1_score": f1,         
            "recall": recall,
            "train_accuracy": train_acc    
        })
        
    run.finish()    

def entrenar_knn(X_train, X_test, y_train, y_test):
    """
    Entrena modelos KNN variando n_neighbors y weights, 
    y sube las curvas de evaluación a Weights & Biases en Runs separados.
    """  
    rango_k = [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51]
    
    tipos_pesos = ['uniform', 'distance']

    tamano_muestra = min(5000, len(X_train))
    indices_aleatorios = np.random.choice(len(X_train), tamano_muestra, replace=False)
    X_train_muestra = X_train[indices_aleatorios]
    y_train_muestra = y_train[indices_aleatorios]
    
    for weight in tipos_pesos:
        print(f"\n Iniciando pruebas para KNN con weights='{weight}'...")
        
        run = wandb.init(
            entity="pd1-c2526-team2",
            project="clasificador-imagenes", 
            name=f"knn-euclidean-{weight}",
            job_type="hyperparameter-tuning",
            config={
                "algoritmo": "KNN",
                "metric": "euclidean",
                "weights": weight
            }
        )

        wandb.define_metric("n_neighbors")
        wandb.define_metric("accuracy", step_metric="n_neighbors")
        wandb.define_metric("f1_score", step_metric="n_neighbors")
        wandb.define_metric("recall", step_metric="n_neighbors")
        
        for k in tqdm(rango_k,desc = "Entrenando Knn con varias Ks"):

            modelo = KNeighborsClassifier(
                n_neighbors=k, 
                weights=weight, 
                metric='euclidean', 
                n_jobs=-1
            )
            
            modelo.fit(X_train, y_train)
            
            y_pred = modelo.predict(X_test)
            y_pred_train = modelo.predict(X_train_muestra)


            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            train_acc = accuracy_score(y_train_muestra, y_pred_train)
            
            print(f"   k={k:2d} | Acc: {acc:.3f} | F1: {f1:.3f} | Recall: {recall:.3f}")
            
            wandb.log({
                "n_neighbors": k,  
                "accuracy": acc,        
                "f1_score": f1,         
                "recall": recall,
                "train_accuracy": train_acc     
            })
            
        run.finish()


def entrenar_random_forest(X_train, X_test, y_train, y_test, profundidad_optima=9):
    """
    Entrena modelos Random Forest variando el número de árboles (n_estimators),
    y sube las métricas de Train y Test a Weights & Biases.
    """    
    run = wandb.init(
        entity="pd1-c2526-team2",
        project="clasificador-imagenes", 
        name="rf-n_estimators-curve",
        job_type="hyperparameter-tuning",
        config={
            "algoritmo": "RandomForest",
            "max_depth_fijo": profundidad_optima,
            "criterion": "entropy"
        }
    )

    wandb.define_metric("n_estimators")
    
    wandb.define_metric("train_accuracy", step_metric="n_estimators")
    wandb.define_metric("accuracy", step_metric="n_estimators") 
    wandb.define_metric("f1_score", step_metric="n_estimators")
    wandb.define_metric("recall", step_metric="n_estimators")
    
    rango_arboles = [10, 25, 50, 100, 150, 200]
    
    tamano_muestra = min(10000, len(X_train)) 
    indices_aleatorios = np.random.choice(len(X_train), tamano_muestra, replace=False)
    X_train_muestra = X_train[indices_aleatorios]
    y_train_muestra = y_train[indices_aleatorios]

    for n in tqdm(rango_arboles,desc="Entrenando RandomForest con distintos números de árboles"):
        
        modelo = RandomForestClassifier(
            n_estimators=n, 
            max_depth=profundidad_optima, 
            criterion="entropy",
            random_state=42, 
            n_jobs=-1 
        )
        
        modelo.fit(X_train, y_train)
        
        y_pred_test = modelo.predict(X_test)
        y_pred_train = modelo.predict(X_train_muestra)
        
        test_acc = accuracy_score(y_test, y_pred_test)
        train_acc = accuracy_score(y_train_muestra, y_pred_train)
        f1 = f1_score(y_test, y_pred_test, average='macro')
        recall = recall_score(y_test, y_pred_test, average='macro')
        
        print(f"   Resultados -> Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f} | F1: {f1:.3f}")
        
        wandb.log({
            "n_estimators": n,  
            "train_accuracy": train_acc,
            "accuracy": test_acc,        
            "f1_score": f1,         
            "recall": recall       
        })
        
    run.finish()

def basline_cat_max(X_train,X_test,y_train,y_test):
    run = wandb.init(entity = "pd1-c2526-team2",
            project="clasificador-imagenes",
            job_type = "train",
            config={
            "algoritmo": "DummyClassifier",
            "estrategia": "most_frequent"
        })

    modelo_baseline = DummyClassifier(strategy='most_frequent')

    modelo_baseline.fit(X_train, y_train)

    ruta_mdl = os.path.join(run.dir, "modelo_baseline.joblib")
    joblib.dump(modelo_baseline, ruta_mdl)

    predicciones = modelo_baseline.predict(X_test)
    predicciones_proba = modelo_baseline.predict_proba(X_test)

    acc = accuracy_score(y_test, predicciones)
    f1 = f1_score(y_test, predicciones, average='macro')

    wandb.log({
        "test_accuracy": acc,
        "test_f1_score": f1
    })

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
    os.environ["WANDB_MODE"] = "online"
    cliente = crear_cliente_minio()
    ruta_ml = "dataset_ml"
    fichero = "embeddings_imagenes.parquet"
    df = bajar_minio(cliente,ruta_ml,fichero)
    X_train,X_test,y_train,y_test = preparar_dataset(df)
    basline_cat_max(X_train,X_test,y_train,y_test)
    entrenar_arbol_decision(X_train,X_test,y_train,y_test,clases)
    entrenar_knn(X_train,X_test,y_train,y_test)
    entrenar_random_forest(X_train,X_test,y_train,y_test)
