from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as predict_router
from app.api.map_routes import router as map_router


app = FastAPI(
    title="MAiDay Model Serving",
    description="API y web para consumir modelos entrenados de src/",
    version="0.1.0",
)

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

app.include_router(predict_router)
app.include_router(map_router)
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


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}
