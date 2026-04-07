import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import nltk
from nltk.corpus import stopwords
import wandb

from utils.funciones_minio import crear_cliente_minio, bajar_minio
from utils.config import PATH_DATASETS_MODELOS
from funciones_texto import bajar_df_texto, x_y_split, train_val_test_split

def optimizar_evaluar_svm(x_train, y_train, x_val, y_val, x_test, y_test):
    print("Bajando stopwords y montando el pipeline...")
    nltk.download('stopwords', quiet=True)
    spanish_stopwords = stopwords.words('spanish')

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words=spanish_stopwords)),
        ("clf", LinearSVC(random_state=42))
    ])

    # Valores a probar. En SVM el parámetro C es el que más nos interesa ajustar
    param_grid = {
        'tfidf__max_features': [5000, 10000, None],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.1, 1.0, 10.0] 
    }

    # Juntamos train y val para el GridSearchCV
    # Con PredefinedSplit nos aseguramos de que no haga cross-validation aleatorio 
    # y use exactamente nuestro x_val para validar
    x_train_val = pd.concat([x_train, x_val])
    y_train_val = pd.concat([y_train, y_val])
    
    # Marcamos train con -1 y val con 0 para que el grid sepa qué es cada cosa
    test_fold = np.concatenate([
        np.full(len(x_train), -1),
        np.full(len(x_val), 0)
    ])
    ps = PredefinedSplit(test_fold)

    print("\nTirando el grid search de SVM...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=ps,                  
        scoring='f1_macro',     # Priorizamos el f1 macro por el desbalanceo
        n_jobs=-1,              
        verbose=1
    )
    
    grid_search.fit(x_train_val, y_train_val)
    
    best_model = grid_search.best_estimator_
    print(f"\nMejores hiperparámetros:\n{grid_search.best_params_}")

    # Pasamos a evaluar el mejor modelo con los datos de test que estaban guardados
    print("\nEvaluando con el conjunto de test...")
    y_pred_test = best_model.predict(x_test)

    test_f1_macro = f1_score(y_test, y_pred_test, average='macro')
    test_precision_macro = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
    test_recall_macro = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
    test_acc = accuracy_score(y_test, y_pred_test)

    # Subimos los resultados limpios a wandb
    run = wandb.init(
        entity="pd1-c2526-team2",
        project="modelo-texto",
        name="svm-optimizado",
        job_type="hyperparameter-search-and-eval",
        group="texto_modelos",
        config={
            "model_type": "svm",
            "search_strategy": "GridSearchCV",
            "best_params": grid_search.best_params_
        }
    )

    wandb.log({
        "test_f1_macro": test_f1_macro,
        "test_precision_macro": test_precision_macro,
        "test_recall_macro": test_recall_macro,
        "test_accuracy": test_acc
    })

    run.finish()
    
    print(f"\nF1-Macro final en Test: {test_f1_macro:.4f}")
    return best_model

if __name__ == "__main__":
    print("Preparando datos...")
    df = bajar_df_texto()
    x, y = x_y_split(df)
    
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y)
    
    optimizar_evaluar_svm(x_train, y_train, x_val, y_val, x_test, y_test)