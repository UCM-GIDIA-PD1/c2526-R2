import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

logger = logging.getLogger("maiday.startup")

# Variables de entorno requeridas para el funcionamiento completo de MAiDay
_REQUIRED_ENV_VARS = {
    "MINIO_ACCESS_KEY": "Credenciales de MinIO (almacenamiento de datos y modelos)",
    "MINIO_SECRET_KEY": "Credenciales de MinIO (almacenamiento de datos y modelos)",
    "WANDB_API_KEY":    "Seguimiento de experimentos con Weights & Biases",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Valida la configuración crítica al arrancar el servidor."""
    missing = [
        (var, desc)
        for var, desc in _REQUIRED_ENV_VARS.items()
        if not os.environ.get(var)
    ]
    if missing:
        logger.warning(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  ⚠  MAiDay — Variables de entorno no configuradas           ║\n"
            "╠══════════════════════════════════════════════════════════════╣"
        )
        for var, desc in missing:
            logger.warning("║  %-20s  %s", var, desc)
        logger.warning(
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║  Pásalas al contenedor con:                                  ║\n"
            "║  podman run --env-file .env --add-host                       ║\n"
            "║    host.containers.internal:host-gateway maiday-app          ║\n"
            "╚══════════════════════════════════════════════════════════════╝"
        )
    else:
        logger.info("✅  Todas las variables de entorno críticas están configuradas.")
    yield

from app.api.routes import router as predict_router
from app.api.map_routes import router as map_router
from app.api.image_routes import router as image_router
from app.api.text_routes import router as text_router
from app.api.game_routes import router as game_router

app = FastAPI(
    title="MAiDay Model Serving",
    description="API y web para consumir modelos entrenados de src/",
    version="0.1.0",
    lifespan=lifespan,
)

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

app.include_router(predict_router)
app.include_router(map_router)
app.include_router(image_router)
app.include_router(text_router)
app.include_router(game_router)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/precios", include_in_schema=False)
def precios() -> FileResponse:
    return FileResponse(WEB_DIR / "precios.html")


@app.get("/analisis", include_in_schema=False)
def analisis() -> FileResponse:
    return FileResponse(WEB_DIR / "analisis.html")


@app.get("/mapas", include_in_schema=False)
def mapas() -> FileResponse:
    return FileResponse(WEB_DIR / "mapas.html")


@app.get("/juego", include_in_schema=False)
def juego() -> FileResponse:
    return FileResponse(WEB_DIR / "juego.html")


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}