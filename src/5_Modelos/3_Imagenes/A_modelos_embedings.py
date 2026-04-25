import pandas as pd
import numpy as np 
import joblib
import wandb
import os
import tempfile
import plotly.express as px
from PIL import Image, ImageOps
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE
import umap
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score,recall_score
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
from utils.funciones_minio import crear_cliente_minio,bajar_minio,subir_minio,buscar_todos_los_archivos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import optuna
from tqdm import tqdm

clases = ["Cocina","Dormitorio","Salón","Banyo"]

class LetterboxPad:
    """
        Clase de transformación que da a las imagenes el formato necesario para pasarlas por Resnet50
    """
    def __init__(self, target_size=224, color=(0, 0, 0)):
        self.target_size = (target_size, target_size)
        self.color = color

    def __call__(self, img):
        return ImageOps.pad(img, self.target_size, color=self.color)

def embeddings_imagenes(cliente, batch_size=32):
    """
        Vectoriza los datos con resnet50 y los sube al minio cuando termina el proceso
    Args:
        cliente (Minio): cliente de minio para bajar las imágenes y subirlas una vez completado
        batch_size (int, optional): cantidad de imagenes que se pasan por la red de una. Defaults to 32.
    """
    print("  Inicio del proceso de transofmración y vectorización de las imágenes con resnet50")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"  Usando {device}")
    pesos = models.ResNet50_Weights.DEFAULT
    modelo_resnet = models.resnet50(weights=pesos)
    modelo_resnet.fc = nn.Identity() 
    modelo_resnet.eval()
    modelo_resnet.to(device)
    
    mis_transforms = transforms.Compose([
        LetterboxPad(target_size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    resultados_finales = []
    
    def procesar_lote(tensores, ids, etiquetas):
        if len(tensores) == 0: return

        batch_tensor = torch.stack(tensores).to(device)
        
        with torch.no_grad():
            embeddings = modelo_resnet(batch_tensor)
            
        embeddings_np = embeddings.cpu().numpy().astype(np.float16)
        
        for i in range(len(ids)):
            resultados_finales.append({
                'id': ids[i],
                'clase': etiquetas[i],
                'embedding': embeddings_np[i]
            })
    
    for clase in clases:
        path = f"cleaned/dataset_vision/{clase}"
        objetos = buscar_todos_los_archivos(cliente,path)
        
        for obj in tqdm(objetos,desc = f"Procesando imagenes de {clase}"):
            
            df_chunk = bajar_minio(cliente,path,obj)
            
            lote_tensores = []
            lote_ids = []
            lote_clases = []
            
            for _, fila in df_chunk.iterrows():
                img = Image.open(io.BytesIO(fila['imagen_bytes'])).convert("RGB")
                tensor_listo = mis_transforms(img)
                    
                lote_tensores.append(tensor_listo)
                lote_ids.append(fila['id'])
                lote_clases.append(clase)
                    
                if len(lote_tensores) == batch_size:
                    procesar_lote(lote_tensores, lote_ids, lote_clases)
                    lote_tensores, lote_ids, lote_clases = [], [], []
            
            procesar_lote(lote_tensores, lote_ids, lote_clases)

    df_final = pd.DataFrame(resultados_finales)
    
    subir_minio(df_final,cliente,"dataset_ml","embeddings_imagenes.parquet")


class ClasificadorEmbeddings(nn.Module):
    """
        Genera el modelo de pytorch con la capa densa para la clasificación
    """
    def __init__(self,num_emb, num_clases=4):
        """
            Inicio de la red neuronal con su configuracion de una capa densa
        Args:
            num_clases (int, optional): Numero de clases que queremos clasificar. Defaults to 4.
        """
        super().__init__()
        self.capa_final = nn.Linear(num_emb, num_clases)

    def forward(self, x):
        """
            Funcion de forward que pasa los datos x recibidos por la ultima capa de embedings de resnet por la capa densa nuestra
        Args:
            x (Tensor): Lista de embedings recibida para una serie de imagenes a clasificar

        Returns:
            Tensor: Valores obtenidos para las 4 distintas clases
        """
        return self.capa_final(x)

def entrenar_mini_red(X_train, y_train, X_test, y_test,nombre_proyecto, num_clases=4, epochs=50, batch_size=256):
    """
        Entrena un algoritmo de softmax con una red neuronal de una sola capa que aprovecha los embedings de
        Resnet 50 para traducirlos a probabilidades de cada clase 
    Args:
        X_train (np.array): conjunto x de train
        y_train (np.array): conjunto y de train
        X_test (np.array): conjunto x de test
        y_test (np.array): conjunto y de test
        num_clases (int, optional): numero de clases de habitaciones. Defaults to 4.
        epochs (int, optional): numero de epocas de entrenamiento. Defaults to 50.
        batch_size (int, optional): cantidad de imagenes que se pasan en un paso del entrenamiento. Defaults to 256.

    Returns:
        nn.Modelo: modelo softmax
    """
    X_train_tensor = torch.tensor(X_train.astype(np.float32))
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.astype(np.float32))
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dimensiones_entrada = X_train.shape[1] 
    
    modelo = ClasificadorEmbeddings(num_emb=dimensiones_entrada, num_clases=num_clases)
    # el criterio crossentropyloss hace su propio softmax internamente por eso no lo metemos antes
    criterio = nn.CrossEntropyLoss() 
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

    run = wandb.init(
        entity="pd1-c2526-team2",
        project=nombre_proyecto,
        name="SoftMax_lastlayer_embeddings",
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
            f1 = f1_score(y_test_tensor.numpy(),predicciones_clases.numpy(), average='macro')
            
        print(f"Epoch [{epoch+1}/{epochs}] | Loss Train: {loss_media_train:.4f} | Loss Test: {loss_test:.4f} | Acc Test: {acc_test*100:.2f}%")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": loss_media_train,
            "test_loss": loss_test,
            "f1_score":f1,
            "accuracy": acc_test
        })
        
    run.finish()

    # metemos la capa de softmax
    modelo_produccion = nn.Sequential(
        modelo,
        nn.Softmax(dim=1)
    )


    return modelo_produccion

def visualizar_umap(df:pd.DataFrame, n_muestras_visualizar=20000):
    """
    Genera una visualización interactiva de los embeddings utilizando el algoritmo UMAP.

    Args:
        df (pd.DataFrame): Dataset que contiene las columnas 'embedding' (vectores) y 'clase' (etiquetas).
        n_muestras_visualizar (int, opcional): Límite máximo de muestras a renderizar para optimizar el rendimiento. Por defecto es 20000.
    """
    
    X = np.stack(df['embedding'].values)
    y = df['clase']  
    codificador = LabelEncoder()

    y_numerico = codificador.fit_transform(y)

    total_datos = len(X)
    porcion_muestrar = min(n_muestras_visualizar / total_datos, 1.0) 
    
    _, X_muestra, _, y_muestra = train_test_split(
        X, y_numerico, test_size=porcion_muestrar, random_state=101, stratify=y_numerico
    )
    nombres_clases = codificador.inverse_transform(y_muestra)

    
    reductor_umap = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        n_components=2, 
        random_state=42,
        n_jobs=-1
    )
    
    X_muestra_2d = reductor_umap.fit_transform(X_muestra)

    df_umap = pd.DataFrame({
        'Dimensión 1': X_muestra_2d[:, 0],
        'Dimensión 2': X_muestra_2d[:, 1],
        'Habitación': nombres_clases
    })
    
    fig = px.scatter(
        df_umap, x='Dimensión 1', y='Dimensión 2', color='Habitación',
        title='Mapa UMAP de Habitaciones',
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    
    fig.update_traces(marker=dict(size=4, line=dict(width=0)))
    fig.update_layout(
        width=1000, height=800, template='plotly_white',
        legend_title_text='Categorías'
    )
    
    fig.show()

def entrenar_svm(X_train, y_train, X_test, y_test,nombre_proyecto,tipo_dataset = ""):
    """
    Optimiza y entrena un modelo Support Vector Machine (LinearSVC) utilizando Optuna y registra las métricas en Weights & Biases.

    Args:
        X_train (np.ndarray): Conjunto de datos de entrenamiento (características).
        y_train (np.ndarray): Etiquetas del conjunto de entrenamiento.
        X_test (np.ndarray): Conjunto de datos de prueba (características).
        y_test (np.ndarray): Etiquetas del conjunto de prueba.
        tipo_dataset (str, opcional): Sufijo para identificar la transformación de los datos en WandB (ej. "_PCA", "_UMAP"). Por defecto es "".

    Returns:
        LinearSVC: El modelo entrenado con los mejores hiperparámetros encontrados.
    """

    run = wandb.init(
        entity="pd1-c2526-team2",
        project=nombre_proyecto,
        name=f"LinearSVC{tipo_dataset}",
        job_type="hyperparameter-tuning"
    )

    def objective(trial):
        c_param = trial.suggest_float('C', 1e-4, 10.0, log=True)
        
        modelo = LinearSVC(
            C=c_param,
            random_state=101,
            max_iter=5000, 
            dual="auto"
        )
        
        metricas = ['accuracy', 'f1_macro', 'recall_macro']
        
        resultados_cv = cross_validate(
            modelo, X_train, y_train, 
            cv=5, 
            scoring=metricas, 
            n_jobs=2
        )
        
        val_acc = resultados_cv['test_accuracy'].mean()
        val_f1 = resultados_cv['test_f1_macro'].mean()
        val_recall = resultados_cv['test_recall_macro'].mean()

        wandb.log({
            "trial": trial.number,
            "accuracy": val_acc,
            "f1_score": val_f1,
            "recall": val_recall,
            "param_C": c_param
        })

        return val_f1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) 

    mejores_params = study.best_params
    print(f" Optuna terminado. Mejores parámetros: {mejores_params}")
    
    mejor_modelo = LinearSVC(**mejores_params, random_state=101, max_iter=5000, dual="auto")
    mejor_modelo.fit(X_train, y_train)
    
    y_pred = mejor_modelo.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    test_recall = recall_score(y_test, y_pred, average='macro')
    
    print(f"  Resultados Finales -> Acc: {test_acc:.3f} | F1: {test_f1:.3f} | Recall: {test_recall:.3f}")
    
    wandb.log({
        "test_final_accuracy": test_acc,
        "test_final_f1_score": test_f1,
        "test_final_recall": test_recall
    })
    
    run.finish()
    
    return mejor_modelo


def visualizar_tsne(df:pd.DataFrame, n_muestras_visualizar=50000):

    """
    Genera una visualización interactiva de los embeddings utilizando PCA seguido del algoritmo t-SNE.

    Args:
        df (pd.DataFrame): Dataset que contiene las columnas 'embedding' (vectores) y 'clase' (etiquetas).
        n_muestras_visualizar (int, opcional): Límite máximo de muestras a renderizar para optimizar el rendimiento. Por defecto es 50000.
    """

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
        title='Mapa T-sne Habitaciones',
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

def entrenar_arbol_decision(X_train, y_train, X_test, y_test,nombre_proyecto,tipo_dataset = ""):
    """
    Optimiza y entrena un modelo de Árbol de Decisión utilizando Optuna y registra las métricas en Weights & Biases.

    Args:
        X_train (np.ndarray): Conjunto de datos de entrenamiento (características).
        y_train (np.ndarray): Etiquetas del conjunto de entrenamiento.
        X_test (np.ndarray): Conjunto de datos de prueba (características).
        y_test (np.ndarray): Etiquetas del conjunto de prueba.
        tipo_dataset (str, opcional): Sufijo para identificar la transformación de los datos en WandB (ej. "_PCA", "_UMAP"). Por defecto es "".

    Returns:
        DecisionTreeClassifier: El modelo entrenado con los mejores hiperparámetros encontrados.
    """
    run = wandb.init(
        entity="pd1-c2526-team2",
        project=nombre_proyecto,
        name=f"DecisionTree_Optuna{tipo_dataset}",
        job_type="hyperparameter-tuning"
    )

    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 5, 50)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        
        modelo = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=101
        )
        
        metricas = ['accuracy', 'f1_macro', 'recall_macro']
        
        resultados_cv = cross_validate(
            modelo, X_train, y_train, 
            cv=5, 
            scoring=metricas, 
            n_jobs=2
        )
        
        val_acc = resultados_cv['test_accuracy'].mean()
        val_f1 = resultados_cv['test_f1_macro'].mean()
        val_recall = resultados_cv['test_recall_macro'].mean()

        wandb.log({
            "trial": trial.number,
            "accuracy": val_acc,
            "f1_score": val_f1,
            "recall": val_recall,
            "param_max_depth": max_depth,
            "param_min_samples_split": min_samples_split,
            "param_min_samples_leaf": min_samples_leaf,
            "param_criterion": criterion
        })

        return val_f1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) 

    mejores_params = study.best_params
    print(f" Optuna terminado. Mejores parámetros: {mejores_params}")
    
    mejor_modelo = DecisionTreeClassifier(**mejores_params, random_state=101)
    mejor_modelo.fit(X_train, y_train)
    
    y_pred = mejor_modelo.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    test_recall = recall_score(y_test, y_pred, average='macro')
    
    print(f"  Resultados Finales -> Acc: {test_acc:.3f} | F1: {test_f1:.3f} | Recall: {test_recall:.3f}")
    
    wandb.log({
        "test_final_accuracy": test_acc,
        "test_final_f1_score": test_f1,
        "test_final_recall": test_recall
    })
    
    run.finish()
    
    return mejor_modelo   

def entrenar_knn(X_train, y_train, X_test, y_test,nombre_proyecto,tipo_dataset = ""):
    """
    Optimiza y entrena un modelo K-Nearest Neighbors (KNN) utilizando Optuna y registra las métricas en Weights & Biases.

    Args:
        X_train (np.ndarray): Conjunto de datos de entrenamiento (características).
        y_train (np.ndarray): Etiquetas del conjunto de entrenamiento.
        X_test (np.ndarray): Conjunto de datos de prueba (características).
        y_test (np.ndarray): Etiquetas del conjunto de prueba.
        tipo_dataset (str, opcional): Sufijo para identificar la transformación de los datos en WandB (ej. "_PCA", "_UMAP"). Por defecto es "".

    Returns:
        KNeighborsClassifier: El modelo entrenado con los mejores hiperparámetros encontrados.
    """
    run = wandb.init(
        entity="pd1-c2526-team2",
        project=nombre_proyecto,
        name=f"KNN{tipo_dataset}",
        job_type="hyperparameter-tuning"
    )

    def objective(trial):
        n_neighbors = trial.suggest_int('n_neighbors', 1, 51) 
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan'])
        
        modelo = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=2
        )
        
        metricas = ['accuracy', 'f1_macro', 'recall_macro']
        
        resultados_cv = cross_validate(
            modelo, X_train, y_train, 
            cv=5, 
            scoring=metricas, 
            n_jobs=-1
        )
        
        val_acc = resultados_cv['test_accuracy'].mean()
        val_f1 = resultados_cv['test_f1_macro'].mean()
        val_recall = resultados_cv['test_recall_macro'].mean()

        wandb.log({
            "trial": trial.number,
            "accuracy": val_acc,
            "f1_score": val_f1,
            "recall": val_recall,
            "param_n_neighbors": n_neighbors,
            "param_weights": weights,
            "param_metric": metric
        })

        return val_f1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) 

    mejores_params = study.best_params
    print(f" Optuna terminado. Mejores parámetros: {mejores_params}")
    
    mejor_modelo = KNeighborsClassifier(**mejores_params, n_jobs=-1)
    mejor_modelo.fit(X_train, y_train)
    
    y_pred = mejor_modelo.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    test_recall = recall_score(y_test, y_pred, average='macro')
    
    print(f"  Resultados Finales -> Acc: {test_acc:.3f} | F1: {test_f1:.3f} | Recall: {test_recall:.3f}")
    
    wandb.log({
        "test_final_accuracy": test_acc,
        "test_final_f1_score": test_f1,
        "test_final_recall": test_recall
    })
    
    run.finish()
    
    return mejor_modelo


def entrenar_random_forest(X_train, y_train, X_test, y_test,nombre_proyecto,tipo_dataset = ""):
    """
    Optimiza y entrena un modelo Random Forest utilizando Optuna y registra las métricas en Weights & Biases.

    Args:
        X_train (np.ndarray): Conjunto de datos de entrenamiento (características).
        y_train (np.ndarray): Etiquetas del conjunto de entrenamiento.
        X_test (np.ndarray): Conjunto de datos de prueba (características).
        y_test (np.ndarray): Etiquetas del conjunto de prueba.
        tipo_dataset (str, opcional): Sufijo para identificar la transformación de los datos en WandB (ej. "_PCA", "_UMAP"). Por defecto es "".

    Returns:
        RandomForestClassifier: El modelo entrenado con los mejores hiperparámetros encontrados.
    """

    run = wandb.init(
        entity="pd1-c2526-team2",
        project=nombre_proyecto,
        name=f"Random_Forest{tipo_dataset}",
        job_type="hyperparameter-tuning"
    )

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
        max_depth = trial.suggest_int('max_depth', 5, 50)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        
        modelo = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion="entropy",
            random_state=101,
            n_jobs=2
        )
        
        metricas = ['accuracy', 'f1_macro', 'recall_macro']
        
        resultados_cv = cross_validate(
            modelo, X_train, y_train, 
            cv=5, 
            scoring=metricas, 
            n_jobs=-1
        )
        
        val_acc = resultados_cv['test_accuracy'].mean()
        val_f1 = resultados_cv['test_f1_macro'].mean()
        val_recall = resultados_cv['test_recall_macro'].mean()

        wandb.log({
            "trial": trial.number,
            "accuracy": val_acc,
            "f1_score": val_f1,
            "recall": val_recall,
            "param_n_estimators": n_estimators,
            "param_max_depth": max_depth
        })

        return val_f1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) 

    mejores_params = study.best_params
    print(f" Optuna terminado. Mejores parámetros: {mejores_params}")
    
    mejor_modelo = RandomForestClassifier(**mejores_params, criterion="entropy", random_state=101)
    mejor_modelo.fit(X_train, y_train)
    
    y_pred = mejor_modelo.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    test_recall = recall_score(y_test, y_pred, average='macro')
    
    print(f"  Resultados Finales -> Acc: {test_acc:.3f} | F1: {test_f1:.3f} | Recall: {test_recall:.3f}")
    
    wandb.log({
        "test_final_accuracy": test_acc,
        "test_final_f1_score": test_f1,
        "test_final_recall": test_recall
    })
    
    run.finish()
    
    return mejor_modelo

def basline_cat_max(X_train,X_test,y_train,y_test,nombre_proyecto):

    """
    Entrena y evalúa un modelo base (baseline) utilizando la estrategia de la clase más frecuente, registrando los resultados en Weights & Biases.

    Args:
        X_train (np.ndarray): Conjunto de datos de entrenamiento (características).
        X_test (np.ndarray): Conjunto de datos de prueba (características).
        y_train (np.ndarray): Etiquetas del conjunto de entrenamiento.
        y_test (np.ndarray): Etiquetas del conjunto de prueba.
    """

    run = wandb.init(entity = "pd1-c2526-team2",
            project=nombre_proyecto,
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)
    print(f"Formato datos de train: {X_train.shape}")
    print(f"Formato datos de test: {X_test.shape}")
    return X_train,X_test,y_train,y_test

def preparar_dataset_PCA(df:pd.DataFrame):
    df = df[df["clase"] != "Comedor"]
    pca = PCA(n_components=512, random_state=101)
    X = np.stack(df['embedding'].values)
    codificador = LabelEncoder()
    y = codificador.fit_transform(df['clase'])    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    varianza_retenida = sum(pca.explained_variance_ratio_)
    print(f"Formato datos de train: {X_train_pca.shape}")
    print(f"Formato datos de test: {X_test_pca.shape}")
    print(f"Información conservada: {varianza_retenida}")
    return X_train_pca,X_test_pca,y_train,y_test

def preparar_dataset_UMAP(df: pd.DataFrame, n_componentes=128):
    """
    Filtra, divide y reduce las dimensiones del dataset usando UMAP.
    """
    df = df[df["clase"] != "Comedor"]
    
    X = np.stack(df['embedding'].values)
    codificador = LabelEncoder()
    y = codificador.fit_transform(df['clase'])    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=101, stratify=y
    )
    
    reductor_umap = umap.UMAP(
        n_components=n_componentes,
        n_neighbors=15,     
        min_dist=0.1,      
        random_state=101, 
        n_jobs=-1          
    )
    
    X_train_umap = reductor_umap.fit_transform(X_train)
    X_test_umap = reductor_umap.transform(X_test)
    
    print(f" Formato datos de train (UMAP): {X_train_umap.shape}")
    print(f" Formato datos de test (UMAP): {X_test_umap.shape}")
    print("Nota: UMAP no calcula 'varianza retenida' al ser una reducción topológica no lineal.")
    
    return X_train_umap, X_test_umap, y_train, y_test

def menu_interactivo():
    
    while True:
        print("==================================================")
        print("      SISTEMA DE CLASIFICACION CON EMBEDDINGS      ")
        print("==================================================")
        print("[1] Cargar los embedings con Resnet")
        print("[2] Descargar los embedings desde minio (Resnet)")
        print("[3] Descargar los embedings PROPIOS (CNN)")
        print("[0] Salir del sistema")
        print("--------------------------------------------------")

        carga = input("Seleccione una opcion: ").strip()

        if carga == "1":
            embeddings_imagenes(crear_cliente_minio())
        elif carga == "0":
            break
        elif carga in ["2", "3"]:
            print("Cargando datos desde MinIO...\n")
    
            try:
                from utils.config import MINIO_EMBEDDINGS
                cliente = crear_cliente_minio()
                
                if carga == "2":
                    fichero = "embeddings_imagenes.parquet"
                    nombre_proyecto = "clasificador-imagenes"
                else:
                    EMBEDDINGS_IMAGENES_PROPIA = "embeddings_cnn_propia.parquet"
                    fichero = EMBEDDINGS_IMAGENES_PROPIA
                    nombre_proyecto = "clasificador-embedings-cnn"
                    
                df = bajar_minio(cliente, MINIO_EMBEDDINGS, fichero)
                print(f"Datos cargados correctamente ({fichero}).")
            except Exception as e:
                print(f"Error al cargar los datos: {e}")
                return
        
            while True:
                print("\n==================================================")
                print("                 MENU PRINCIPAL                   ")
                print("==================================================")
                print("[1] Visualizar datos")
                print("[2] Entrenar modelos")
                print("[0] Salir del sistema")
                print("--------------------------------------------------")
                
                opcion_principal = input("Seleccione una opcion: ").strip()
                
                if opcion_principal == "0":
                    print("\nSaliendo del sistema.")
                    break
                    
                elif opcion_principal == "1":
                    while True:
                        print("\n--- MENU DE VISUALIZACION ---")
                        print("[1] Visualizar con t-SNE")
                        print("[2] Visualizar con UMAP")
                        print("[0] Volver al menu principal")
                        
                        op_vis = input("Seleccione metodo: ").strip()
                        if op_vis == "0":
                            break
                        elif op_vis == "1":
                            print("\nGenerando visualizacion t-SNE...")
                            visualizar_tsne(df)
                        elif op_vis == "2":
                            print("\nGenerando visualizacion UMAP...")
                            visualizar_umap(df)
                        else:
                            print("Opcion no valida.")
                            
                elif opcion_principal == "2":
                    while True:
                        print("\n--- PREPARACION DEL DATASET ---")
                        print("[1] Usar Embeddings enteros (Original)")
                        print("[2] Reducir dimensiones con PCA")
                        print("[3] Reducir dimensiones con UMAP")
                        print("[0] Volver al menu principal")
                        
                        op_data = input("Seleccione transformacion de datos: ").strip()
                        
                        if op_data == "0":
                            break
                        
                        tipo_dataset = ""
                        X_train, X_test, y_train, y_test = None, None, None, None
                        
                        if op_data == "1":
                            print("\nPreparando embeddings originales...")
                            X_train, X_test, y_train, y_test = preparar_dataset(df)
                            tipo_dataset = ""
                        elif op_data == "2":
                            print("\nPreparando reducción con PCA...")
                            X_train, X_test, y_train, y_test = preparar_dataset_PCA(df)
                            tipo_dataset = "_PCA"
                        elif op_data == "3":
                            print("\nPreparando reducción con UMAP...")
                            X_train, X_test, y_train, y_test = preparar_dataset_UMAP(df)
                            tipo_dataset = "_UMAP"
                        else:
                            print("Opcion no valida.")
                            continue
                        
                        while True:
                            print(f"\n--- ENTRENAMIENTO DE MODELOS (Dataset actual: {tipo_dataset if tipo_dataset else 'Original'}) ---")
                            print("[1] Modelo SVM (LinearSVC)")
                            print("[2] Modelo KNN")
                            print("[3] Modelo Random Forest")
                            print("[4] Modelo Decision Tree")
                            if op_data == "1":
                                print("[5] Mini Red Neuronal (SoftMax)")
                            print("[0] Volver a la seleccion de dataset")
                            
                            op_mod = input("Seleccione el modelo a entrenar: ").strip()
                            
                            if op_mod == "0":
                                break
                            elif op_mod == "1":
                                entrenar_svm(X_train, y_train, X_test, y_test,nombre_proyecto, tipo_dataset)
                            elif op_mod == "2":
                                entrenar_knn(X_train, y_train, X_test, y_test,nombre_proyecto, tipo_dataset)
                            elif op_mod == "3":
                                entrenar_random_forest(X_train, y_train, X_test, y_test,nombre_proyecto, tipo_dataset)
                            elif op_mod == "4":
                                entrenar_arbol_decision(X_train, y_train, X_test, y_test,nombre_proyecto, tipo_dataset)
                            elif op_mod == "5":
                                if op_data == "1":
                                    entrenar_mini_red(X_train, y_train, X_test, y_test,nombre_proyecto)
                                else:
                                    print("\n[ERROR] La Mini Red Neuronal solo admite embeddings originales. No compatible con PCA/UMAP.")
                            else:
                                print("Opcion no valida.")
                            
                            print("\nEntrenamiento finalizado. Se pueden ver los resultados en Weights & Biases.")
                else:
                    print("Opcion no valida. Por favor, introduzca 0, 1 o 2.")

if __name__ == "__main__":
    menu_interactivo()