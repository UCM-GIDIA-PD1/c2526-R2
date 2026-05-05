import os

import joblib
import numpy as np
import wandb


class TextPredictor:
    def __init__(self):
        self.model = None
        # Hiperparámetros del mejor modelo (referencia, ya quedan dentro del .joblib):
        #   vectorizer: tfidf
        #   max_df: 0.9634328488868874
        #   min_df: 4
        #   max_features: 14000
        #   ngram_range: (1, 1)
        #   C (LinearSVC / SVC): 0.5101843757043346
        # Apunta al artifact (no al run). Usar ':latest' para coger siempre
        # la última versión subida; cambia a ':v0', ':v1', etc. si quieres pinear.
        self.wandb_artifact_path = "pd1-c2526-team2/modelo-texto-final/mejor-svm-texto:latest"
        self.model_name = "mejor_SVM_texto.joblib"

        self.model_cache_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "model_cache"
        )
        os.makedirs(self.model_cache_dir, exist_ok=True)

    def load_model_if_needed(self):
        if self.model is not None:
            return

        print(f"Descargando artifact {self.wandb_artifact_path} desde W&B...")
        api = wandb.Api()
        artifact = api.artifact(self.wandb_artifact_path)
        artifact_dir = artifact.download(root=self.model_cache_dir)
        local_model_path = os.path.join(artifact_dir, self.model_name)
        print("Descarga completada.")

        print("Cargando modelo en memoria...")
        self.model = joblib.load(local_model_path)
        print("Modelo cargado correctamente.")

    def predict(self, texto: str) -> dict:
        """
        Recibe un string de texto, lo pasa por el pipeline (TF-IDF + SVM)
        y devuelve la clase predicha junto con probabilidades o scores
        de decisión, según lo que exponga el modelo.
        """
        self.load_model_if_needed()

        prediccion = self.model.predict([texto])[0]
        resultado = {"clase": str(prediccion)}

        # Si el SVM se entrenó con probability=True (SVC) tendrá predict_proba.
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba([texto])[0]
            clases = self.model.classes_
            resultado["probabilidades"] = {
                str(clases[i]): float(probas[i]) for i in range(len(clases))
            }
        # LinearSVC y SVC sin probability=True exponen decision_function.
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function([texto])[0]
            clases = self.model.classes_
            scores_arr = np.atleast_1d(scores)
            if len(clases) == 2 and scores_arr.shape == (1,):
                # Caso binario: decision_function devuelve un único score
                resultado["scores"] = {
                    str(clases[1]): float(scores_arr[0]),
                    str(clases[0]): float(-scores_arr[0]),
                }
            else:
                resultado["scores"] = {
                    str(clases[i]): float(scores_arr[i])
                    for i in range(len(clases))
                }

        return resultado

    def predict_batch(self, textos: list[str]) -> list[dict]:
        """Versión batch para clasificar varios textos a la vez."""
        self.load_model_if_needed()

        predicciones = self.model.predict(textos)
        resultados = [{"clase": str(p)} for p in predicciones]

        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(textos)
            clases = self.model.classes_
            for i, prob_array in enumerate(probas):
                resultados[i]["probabilidades"] = {
                    str(clases[j]): float(prob_array[j])
                    for j in range(len(clases))
                }
        elif hasattr(self.model, "decision_function"):
            scores_all = np.atleast_2d(self.model.decision_function(textos))
            clases = self.model.classes_
            for i in range(len(textos)):
                row = scores_all[i]
                if len(clases) == 2 and row.shape == (1,):
                    resultados[i]["scores"] = {
                        str(clases[1]): float(row[0]),
                        str(clases[0]): float(-row[0]),
                    }
                else:
                    resultados[i]["scores"] = {
                        str(clases[j]): float(row[j])
                        for j in range(len(clases))
                    }

        return resultados


# Instancia singleton para que el router la importe (mismo patrón que image_predictor).
predictor = TextPredictor()