from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from app.services.model_loader import model_loader


def _normalize_result(result: Any) -> str | float | int:
    if isinstance(result, (np.ndarray, list, tuple)):
        first = result[0]
        if isinstance(first, np.generic):
            return first.item()
        return first
    if isinstance(result, np.generic):
        return result.item()
    return result


def predict_tabular(model_key: str, payload: dict[str, Any]) -> str | float | int:
    model = model_loader.get(model_key)
    df = pd.DataFrame([payload])
    
    # Preprocesamiento de inferencia consistente con el entrenamiento
    cat_cols = df.select_dtypes(exclude=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    if cat_cols:
        df[cat_cols] = df[cat_cols].fillna('Desconocido').astype(str)
        
    prediction = model.predict(df)
    return _normalize_result(prediction)


def predict_text(payload: str) -> str | float | int:
    model = model_loader.get("texto")
    prediction = model.predict([payload])
    return _normalize_result(prediction)


def predict_image(image_bytes: bytes) -> str | float | int:
    model = model_loader.get("imagen")
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Generic preprocessing to support common sklearn/keras style pipelines.
    image_arr = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0
    batch = np.expand_dims(image_arr, axis=0)

    prediction = model.predict(batch)
    return _normalize_result(prediction)
