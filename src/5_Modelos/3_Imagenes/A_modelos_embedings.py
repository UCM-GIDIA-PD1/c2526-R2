import pandas as pd
import numpy as np 
import joblib
import wandb
import os
import tempfile
import plotly.express as px
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score,recall_score
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
from utils.funciones_minio import crear_cliente_minio,bajar_minio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

clases = ['Cocina', 'Dormitorio', 'Salón', 'Banyo']


class ClasificadorEmbeddings(nn.Module):
    def __init__(self, num_clases=4):
        super().__init__()
        self.capa_final = nn.Linear(2048, num_clases)

    def forward(self, x):
        return self.capa_final(x)

def entrenar_mini_red(X_train, y_train, X_test, y_test, num_clases=4, epochs=50, batch_size=256):
    
    X_train_tensor = torch.tensor(X_train.astype(np.float32))
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.astype(np.float32))
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    modelo = ClasificadorEmbeddings(num_clases=num_clases)
    criterio = nn.CrossEntropyLoss() 
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

    run = wandb.init(
        entity="pd1-c2526-team2",
        project="clasificador-imagenes",
        name="pytorch-embeddings-nn_linear",
        job_type="train",
        config={"epochs": epochs, "batch_size": batch_size, "lr": 0.001}
    )
    
    
    for epoch in range(epochs):
        modelo.train()
        loss_total_train = 0
        
        for batch_X, batch_y in loader_train:
            optimizador.zero_grad()          
            predicciones = modelo(batch_X)  
            loss = criterio(predicciones, batch_y) 
            loss.backward()                   
            optimizador.step()              
            
            loss_total_train += loss.item()
            
        loss_media_train = loss_total_train / len(loader_train)
        
        modelo.eval()
        with torch.no_grad(): 
            predicciones_test = modelo(X_test_tensor)
            loss_test = criterio(predicciones_test, y_test_tensor).item()
            
            _, predicciones_clases = torch.max(predicciones_test, 1)
            
            acc_test = accuracy_score(y_test_tensor.numpy(), predicciones_clases.numpy())
            
        print(f"Epoch [{epoch+1}/{epochs}] | Loss Train: {loss_media_train:.4f} | Loss Test: {loss_test:.4f} | Acc Test: {acc_test*100:.2f}%")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": loss_media_train,
            "test_loss": loss_test,
            "accuracy": acc_test
        })
        
    run.finish()
    return modelo

def visualizar_tsne(df:pd.DataFrame, n_muestras_visualizar=50000): 
    X = np.stack(df['embedding'].values)
    y = df['clase']  
    total_datos = len(X)
    porcion_muestra = min(n_muestras_visualizar / total_datos, 1.0) 

    codificador = LabelEncoder()

    y_numerico = codificador.fit_transform(y)

    _, X_muestra, _, y_muestra = train_test_split(
        X, y_numerico, 
        test_size=porcion_muestra, 
        random_state=101, 
        stratify=y_numerico
    )
    
    nombres_clases = codificador.inverse_transform(y_muestra)
    print(f" Muestra de {X_muestra.shape} extraída.")

    pca = PCA(n_components=50, random_state=101)
    X_muestra_pca = pca.fit_transform(X_muestra)

    tsne = TSNE(
        n_components=2, 
        perplexity=30, 
        n_jobs=-1, 
        random_state=101, 
        verbose=1
    )
    X_muestra_2d = tsne.fit_transform(X_muestra_pca)

    
    df_tsne = pd.DataFrame({
        'Dimensión 1': X_muestra_2d[:, 0],
        'Dimensión 2': X_muestra_2d[:, 1],
        'Habitación': nombres_clases
    })
    
    fig = px.scatter(
        df_tsne,
        x='Dimensión 1',
        y='Dimensión 2',
        color='Habitación',
        title='Mapa Habitaciones',
        opacity=0.6, 
        color_discrete_sequence=px.colors.qualitative.Bold 
    )
    
    fig.update_traces(marker=dict(size=5, line=dict(width=0))) 
    fig.update_layout(
        width=1000,
        height=800,
        template='plotly_white',
        legend_title_text='Categorías',
        legend=dict(
            itemsizing='constant', 
            font=dict(size=14)
        )
    )
    
    fig.show()

def entrenar_arbol_decision(X_train, X_test, y_train, y_test, etiquetas_clases, max_profundidad=40):
    """
    Entrena un DecisionTreeClassifier, evalúa sus métricas y 
    sube tanto los resultados como el modelo físico a Weights & Biases.
    """
    
    run = wandb.init(
        entity = "pd1-c2526-team2",
        project="clasificador-imagenes", 
        name="Decision_Tree_PCA",
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

        print(f"   Profundidad: {p} | Acc: {acc:.3f} | F1: {f1:.3f} | Recall: {recall:.3f} | Train Acc: {train_acc:.3f}")
        
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
            name=f"knn-euclidean-{weight}_PCA",
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
        name="rf-n_estimators-curve_PCA",
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
    df = df[df["clase"] != "Comedor"]
    X = np.stack(df['embedding'].values)
    codificador = LabelEncoder()
    y = codificador.fit_transform(df['clase'])  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, stratify=y)
    print(f"Formato datos de train: {X_train.shape}")
    print(f"Formato datos de test: {X_test.shape}")
    return X_train,X_test,y_train,y_test

def preparar_dataset_PCA(df:pd.DataFrame):
    df = df[df["clase"] != "Comedor"]
    pca = PCA(n_components=512, random_state=42)
    X = np.stack(df['embedding'].values)
    codificador = LabelEncoder()
    y = codificador.fit_transform(df['clase'])    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, stratify=y)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    varianza_retenida = sum(pca.explained_variance_ratio_)
    print(f"Formato datos de train: {X_train_pca.shape}")
    print(f"Formato datos de test: {X_test_pca.shape}")
    print(f"Información conservada: {varianza_retenida}")
    return X_train_pca,X_test_pca,y_train,y_test

if __name__ == "__main__":
    cliente = crear_cliente_minio()
    ruta_ml = "dataset_ml"
    fichero = "embeddings_imagenes.parquet"
    df = bajar_minio(cliente,ruta_ml,fichero)
    visualizar_tsne(df)