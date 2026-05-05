import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas import TextoInput
from app.services.text_predictor import predictor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predictions"])


class TextPredictionResponse(BaseModel):
    clase: str
    scores: dict[str, float] | None = None
    probabilidades: dict[str, float] | None = None


@router.post("/texto", response_model=TextPredictionResponse)
def predict_texto(data: TextoInput):
    """
    Endpoint para clasificar la descripción textual de un anuncio
    (particular / promotora / intermediario).
    """
    if not data.texto or not data.texto.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío.")

    try:
        resultado = predictor.predict(data.texto)
        return TextPredictionResponse(
            clase=resultado["clase"],
            scores=resultado.get("scores"),
            probabilidades=resultado.get("probabilidades"),
        )
    except Exception as e:
        logger.exception("Error en predicción de texto")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno procesando el texto: {e}",
        )