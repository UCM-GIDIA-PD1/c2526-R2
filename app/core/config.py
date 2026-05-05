import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

# ── Detección de entorno contenedor ─────────────────────────────────────────────
# Se considera que la app corre dentro de un contenedor si existe /.dockerenv
# (creado por Docker/Podman) o si la variable de entorno IS_CONTAINER=true.
IS_CONTAINER: bool = (
    Path("/.dockerenv").exists()
    or os.environ.get("IS_CONTAINER", "").lower() == "true"
)

# ── Ajuste automático del endpoint de MinIO ──────────────────────────────────────
# Dentro de un contenedor, 'localhost' apunta al propio contenedor, no a la máquina
# host. Si se detecta un entorno contenedor con endpoint local, se redirige
# automáticamente a 'host.containers.internal' (válido para Podman y Docker en
# Windows/macOS con --add-host host.containers.internal:host-gateway).
_minio_endpoint = os.environ.get("MINIO_ENDPOINT", "")
if IS_CONTAINER and _minio_endpoint in ("localhost", "127.0.0.1"):
    MINIO_ENDPOINT = "host.containers.internal"
else:
    MINIO_ENDPOINT = _minio_endpoint

# ── Rutas de modelos ─────────────────────────────────────────────────────────────
# Los modelos entrenados se almacenan en model_cache/ en la raíz del proyecto.
# Esta carpeta se copia durante el build (no está en .containerignore).
# Para sobreescribir en producción, define las variables de entorno correspondientes.
DEFAULT_VENTA_MODEL_PATH = ROOT_DIR / "model_cache" / "venta_model.pkl"
DEFAULT_ALQUILER_MODEL_PATH = ROOT_DIR / "model_cache" / "alquiler_model.pkl"
DEFAULT_TEXTO_MODEL_PATH = ROOT_DIR / "model_cache" / "texto_model.pkl"
DEFAULT_IMAGEN_MODEL_PATH = ROOT_DIR / "model_cache" / "imagen_model.pkl"
