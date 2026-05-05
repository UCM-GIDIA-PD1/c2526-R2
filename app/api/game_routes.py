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
    calcular_ganador_comparacion,
    cargar_vivienda_aleatoria,
    cargar_dos_viviendas_aleatorias,
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


class DosViviendasResponse(BaseModel):
    """Respuesta con los datos de dos viviendas para comparar."""
    vivienda1: ViviendaJuegoResponse
    vivienda2: ViviendaJuegoResponse


class ResultadoComparacionInput(BaseModel):
    """Datos que envía el cliente para calcular el resultado de la comparación."""
    eleccion_usuario: int = Field(description="Elección del usuario (1 o 2).")
    vivienda1_precio_real: float
    vivienda2_precio_real: float
    vivienda1_features: dict
    vivienda2_features: dict


class ResultadoComparacionResponse(BaseModel):
    """Resultado completo de la comparación."""
    vivienda1_precio_real: float
    vivienda2_precio_real: float
    vivienda1_precio_modelo: float
    vivienda2_precio_modelo: float
    mas_caro_real: int
    eleccion_usuario: int
    eleccion_modelo: int
    ganador: str
    mensaje: str


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


@router.get("/dos_viviendas", response_model=DosViviendasResponse)
def obtener_dos_viviendas_juego() -> DosViviendasResponse:
    """
    Selecciona un barrio aleatorio y dentro de él dos viviendas aleatorias.
    Devuelve las pistas e imágenes de ambas.
    """
    try:
        f1, p1, img1, f2, p2, img2 = cargar_dos_viviendas_aleatorias()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error cargando dos viviendas aleatorias")
        raise HTTPException(status_code=500, detail=f"Error al cargar viviendas: {e}")

    img1_b64, img1_mime = None, None
    if img1:
        img1_b64 = base64.b64encode(img1).decode("utf-8")
        img1_mime = _detectar_mime(img1)

    img2_b64, img2_mime = None, None
    if img2:
        img2_b64 = base64.b64encode(img2).decode("utf-8")
        img2_mime = _detectar_mime(img2)

    v1 = ViviendaJuegoResponse(
        pistas=extraer_pistas(f1),
        imagen_b64=img1_b64,
        imagen_mime=img1_mime,
        features_para_modelo=f1,
        precio_real=p1,
    )
    v2 = ViviendaJuegoResponse(
        pistas=extraer_pistas(f2),
        imagen_b64=img2_b64,
        imagen_mime=img2_mime,
        features_para_modelo=f2,
        precio_real=p2,
    )

    return DosViviendasResponse(vivienda1=v1, vivienda2=v2)


@router.post("/resultado_comparacion", response_model=ResultadoComparacionResponse)
def calcular_resultado_comparacion(data: ResultadoComparacionInput) -> ResultadoComparacionResponse:
    """
    Recibe la elección del usuario (1 o 2), obtiene las predicciones del modelo para ambas,
    y determina el ganador del modo 'Cuál es el más caro'.
    """
    try:
        precio_modelo_1 = predecir_modelo(data.vivienda1_features)
        precio_modelo_2 = predecir_modelo(data.vivienda2_features)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado al predecir con el modelo")
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")

    resultado = calcular_ganador_comparacion(
        eleccion_usuario=data.eleccion_usuario,
        vivienda1_precio_real=data.vivienda1_precio_real,
        vivienda2_precio_real=data.vivienda2_precio_real,
        vivienda1_precio_modelo=precio_modelo_1,
        vivienda2_precio_modelo=precio_modelo_2,
    )
    return ResultadoComparacionResponse(**resultado)
