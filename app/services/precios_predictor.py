import os
import joblib
import numpy as np
import pandas as pd
import wandb


class PreciosPredictor:
    def __init__(self):
        self.models = {}
        
        self.wandb_artifact_paths = {
            "venta": "pd1-c2526-team2/modelo-precio-viviendas-venta/xgboost-hibrido-venta:latest",
            "alquiler": "pd1-c2526-team2/modelo-precio-viviendas-alquiler/xgboost-hibrido-alquiler:latest"
        }
        
        self.model_names = {
            "venta": "modelo_produccion_venta.pkl",
            "alquiler": "modelo_produccion_alquiler.pkl"
        }

        self.model_cache_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "model_cache"
        )
        os.makedirs(self.model_cache_dir, exist_ok=True)

    def load_model_if_needed(self, model_key: str):
        if model_key in self.models and self.models[model_key] is not None:
            return

        wandb_path = self.wandb_artifact_paths[model_key]
        model_name = self.model_names[model_key]

        print(f"Descargando artifact {wandb_path} desde W&B...")
        api = wandb.Api()
        artifact = api.artifact(wandb_path)
        artifact_dir = artifact.download(root=self.model_cache_dir)
        local_model_path = os.path.join(artifact_dir, model_name)
        print("Descarga completada.")

        print(f"Cargando modelo de {model_key} en memoria...")
        self.models[model_key] = joblib.load(local_model_path)
        print(f"Modelo de {model_key} cargado correctamente.")

    def _normalize_result(self, result) -> float:
        if isinstance(result, (np.ndarray, list, tuple)):
            first = result[0]
            if isinstance(first, np.generic):
                return float(first.item())
            return float(first)
        if isinstance(result, np.generic):
            return float(result.item())
        return float(result)

    def predict(self, model_key: str, payload: dict) -> float:
        """
        Recibe el tipo de modelo ('venta' o 'alquiler') y los datos,
        lo procesa y devuelve la predicción.
        """
        if model_key not in ["venta", "alquiler"]:
            raise ValueError(f"Modelo '{model_key}' no soportado.")
            
        self.load_model_if_needed(model_key)
        model = self.models[model_key]

        df = pd.DataFrame([payload])
        
        # Extraer las columnas exactas del preprocesador del modelo
        preprocessor = model.named_steps.get('preprocessor')
        if preprocessor:
            num_cols = preprocessor.transformers_[0][2]
            cat_cols = preprocessor.transformers_[1][2]
            
            # Asegurar que existan todas las columnas que espera el modelo
            for c in num_cols + cat_cols:
                if c not in df.columns:
                    df[c] = np.nan
            
            cat_cols_df = [c for c in cat_cols if c in df.columns]
            num_cols_df = [c for c in num_cols if c in df.columns]
            
            if cat_cols_df:
                df[cat_cols_df] = df[cat_cols_df].fillna('Desconocido').astype(str).replace(r'\.0$', '', regex=True)
            if num_cols_df:
                df[num_cols_df] = df[num_cols_df].astype(float)
        else:
            # Fallback
            cat_cols = df.select_dtypes(exclude=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            if cat_cols:
                df[cat_cols] = df[cat_cols].fillna('Desconocido').astype(str).replace(r'\.0$', '', regex=True)
            
        prediction = model.predict(df)
        return self._normalize_result(prediction)


# Instancia singleton para que el router la importe
predictor = PreciosPredictor()
