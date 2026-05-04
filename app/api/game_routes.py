"""
game_routes.py

Endpoints FastAPI para el minijuego "¿Conoces Madrid mejor que la IA?".

Endpoints:
    GET  /juego/vivienda   — Barrio + vivienda aleatorios; devuelve pistas e imagen
    POST /juego/resultado  — Recibe precio del usuario; compara con modelo y precio real
"""

import base64
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.game_service import (
    calcular_ganador,
    cargar_vivienda_aleatoria,
    extraer_pistas,
    predecir_modelo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/juego", tags=["minijuego"])


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ViviendaJuegoResponse(BaseModel):
    """Respuesta con los datos de una vivienda para el juego."""
    pistas: dict
    imagen_b64: str | None = Field(
        None, description="Imagen en Base64 (WebP/JPEG/PNG)."
    )
    imagen_mime: str | None = Field(
        None, description="MIME type de la imagen."
    )
    features_para_modelo: dict = Field(
        description="Features necesarias para que el modelo prediga."
    )
    precio_real: float = Field(
        description="Precio real (se devuelve al cliente para la validación final)."
    )


class ResultadoJuegoInput(BaseModel):
    """Datos que envía el cliente para calcular el resultado."""
    precio_usuario: float = Field(gt=0, description="Precio adivinado por el usuario (€).")
    precio_real:    float = Field(gt=0, description="Precio real de la vivienda (€).")
    features_para_modelo: dict = Field(description="Features de la vivienda para el modelo.")


class ResultadoJuegoResponse(BaseModel):
    """Resultado completo del juego con comparativa usuario vs modelo."""
    precio_real:       float
    precio_usuario:    float
    precio_modelo:     float
    error_usuario:     float
    error_modelo:      float
    pct_error_usuario: float
    pct_error_modelo:  float
    ganador:           str
    diferencia:        float
    mensaje:           str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

def _detectar_mime(imagen_bytes: bytes) -> str:
    """
    Detecta el MIME type de la imagen por magic bytes.
    El scraper guarda SIEMPRE en WebP → es el default si no reconocemos otra cosa.
    WebP: bytes 0-3 = RIFF, bytes 8-11 = WEBP
    """
    if len(imagen_bytes) >= 12:
        if imagen_bytes[:4] == b"RIFF" and imagen_bytes[8:12] == b"WEBP":
            return "image/webp"
    if imagen_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if imagen_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    # Fallback: todas las imágenes del scraper son WebP
    return "image/webp"


@router.get("/vivienda", response_model=ViviendaJuegoResponse)
def obtener_vivienda_juego() -> ViviendaJuegoResponse:
    """
    Selecciona un barrio aleatorio y dentro de él una vivienda aleatoria.
    Devuelve: pistas visuales, imagen en Base64, features para el modelo y precio real.
    """
    try:
        features, precio_real, imagen_bytes = cargar_vivienda_aleatoria()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error cargando vivienda aleatoria")
        raise HTTPException(status_code=500, detail=f"Error al cargar vivienda: {e}")

    # Codificar imagen en Base64 si está disponible
    imagen_b64  = None
    imagen_mime = None
    if imagen_bytes:
        imagen_b64  = base64.b64encode(imagen_bytes).decode("utf-8")
        imagen_mime = _detectar_mime(imagen_bytes)

    return ViviendaJuegoResponse(
        pistas=extraer_pistas(features),
        imagen_b64=imagen_b64,
        imagen_mime=imagen_mime,
        features_para_modelo=features,
        precio_real=precio_real,
    )


@router.post("/resultado", response_model=ResultadoJuegoResponse)
def calcular_resultado_juego(data: ResultadoJuegoInput) -> ResultadoJuegoResponse:
    """
    Recibe la predicción del usuario, obtiene la del modelo y determina el ganador.
    """
    try:
        precio_modelo = predecir_modelo(data.features_para_modelo)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado al predecir con el modelo")
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")

    resultado = calcular_ganador(
        precio_usuario=data.precio_usuario,
        precio_modelo=precio_modelo,
        precio_real=data.precio_real,
    )
    return ResultadoJuegoResponse(**resultado)
