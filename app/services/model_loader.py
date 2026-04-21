import os
import pickle
from pathlib import Path
from typing import Any

from app.core.config import (
    DEFAULT_ALQUILER_MODEL_PATH,
    DEFAULT_IMAGEN_MODEL_PATH,
    DEFAULT_TEXTO_MODEL_PATH,
    DEFAULT_VENTA_MODEL_PATH,
)


class ModelLoader:
    """Lazy model loader with in-memory cache."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._paths: dict[str, Path] = {
            "venta": Path(os.getenv("MODEL_VENTA_PATH", DEFAULT_VENTA_MODEL_PATH)),
            "alquiler": Path(
                os.getenv("MODEL_ALQUILER_PATH", DEFAULT_ALQUILER_MODEL_PATH)
            ),
            "texto": Path(os.getenv("MODEL_TEXTO_PATH", DEFAULT_TEXTO_MODEL_PATH)),
            "imagen": Path(os.getenv("MODEL_IMAGEN_PATH", DEFAULT_IMAGEN_MODEL_PATH)),
        }

    def get(self, model_key: str) -> Any:
        if model_key in self._cache:
            return self._cache[model_key]

        model_path = self._paths[model_key]
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontro el artefacto del modelo '{model_key}': {model_path}"
            )

        with model_path.open("rb") as file:
            model = pickle.load(file)

        self._cache[model_key] = model
        return model

    def get_path(self, model_key: str) -> Path:
        return self._paths[model_key]


model_loader = ModelLoader()
