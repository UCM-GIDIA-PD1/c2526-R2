import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import nltk
from nltk.corpus import stopwords
import wandb

from funciones_texto import bajar_df_texto, x_y_split, train_val_test_split

def optimizar_evaluar_naive_bayes(x_train, y_train, x_val, y_val, x_test, y_test):
    print("Descargando stopwords y configurando modelo...")
    nltk.download('stopwords', quiet=True)
    spanish_stopwords = stopwords.words('spanish')

    # 1. Definir el pipeline base
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words=spanish_stopwords)),
        ("clf", MultinomialNB())
    ])

    # 2. Definir el espacio de búsqueda (Grid)
    param_grid = {
        'tfidf__max_features': [5000, 10000, None],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__alpha': [0.1, 0.5, 1.0, 5.0]
    }

    # 3. Configurar PredefinedSplit para forzar el uso de x_val
    x_train_val = pd.concat([x_train, x_val])
    y_train_val = pd.concat([y_train, y_val])
    
    test_fold = np.concatenate([
        np.full(len(x_train), -1),
        np.full(len(x_val), 0)
    ])
    ps = PredefinedSplit(test_fold)

    # 4. Configurar y ejecutar la búsqueda de hiperparámetros
    print("\nIniciando búsqueda de hiperparámetros con GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=ps,                  
        scoring='f1_macro',
        n_jobs=-1,              
        verbose=1
    )
    
    grid_search.fit(x_train_val, y_train_val)
    
    best_model = grid_search.best_estimator_
    print(f"\n✅ Mejores hiperparámetros encontrados:\n{grid_search.best_params_}")

    # 5. Evaluar el MEJOR modelo sobre el conjunto de TEST
    print("\nEvaluando el mejor modelo en el conjunto de TEST...")
    y_pred_test = best_model.predict(x_test)

    # Calculamos las métricas
    test_f1_macro = f1_score(y_test, y_pred_test, average='macro')
    test_precision_macro = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
    test_recall_macro = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
    test_acc = accuracy_score(y_test, y_pred_test)

    # 6. Registrar en Weights & Biases (Sin la matriz de confusión)
    run = wandb.init(
        entity="pd1-c2526-team2",
        project="modelo-texto",
        name="naive-bayes-optimizado",
        job_type="hyperparameter-search-and-eval",
        group="texto_modelos",
        config={
            "model_type": "naive-bayes",
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
    
    print(f"\n📊 Resultados Finales (Test) - F1-Macro: {test_f1_macro:.4f}")
    return best_model

if __name__ == "__main__":
    df = bajar_df_texto()
    x, y = x_y_split(df)
    
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y)
    
    optimizar_evaluar_naive_bayes(x_train, y_train, x_val, y_val, x_test, y_test)