from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

# Default artifact locations inside src/. Override with env vars if needed.
DEFAULT_VENTA_MODEL_PATH = SRC_DIR / "model_artifacts" / "venta_model.pkl"
DEFAULT_ALQUILER_MODEL_PATH = SRC_DIR / "model_artifacts" / "alquiler_model.pkl"
DEFAULT_TEXTO_MODEL_PATH = SRC_DIR / "model_artifacts" / "texto_model.pkl"
DEFAULT_IMAGEN_MODEL_PATH = SRC_DIR / "model_artifacts" / "imagen_model.pkl"
