from fastapi import APIRouter, File, UploadFile

from app.schemas import AlquilerInput, PredictionResponse, TextoInput, VentaInput
from app.services.demo_predictors import (
    predict_alquiler_demo,
    predict_imagen_demo,
    predict_texto_demo,
    predict_venta_demo,
)


router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post("/venta", response_model=PredictionResponse)
def predict_venta(data: VentaInput) -> PredictionResponse:
    prediction = predict_venta_demo(
        m2=data.m2,
        habitaciones=data.habitaciones,
        banos=data.banos,
        codigo_postal=data.codigo_postal,
    )
    return PredictionResponse(model_name="venta-demo", prediction=prediction)


@router.post("/alquiler", response_model=PredictionResponse)
def predict_alquiler(data: AlquilerInput) -> PredictionResponse:
    prediction = predict_alquiler_demo(
        m2=data.m2,
        habitaciones=data.habitaciones,
        banos=data.banos,
        codigo_postal=data.codigo_postal,
    )
    return PredictionResponse(model_name="alquiler-demo", prediction=prediction)


@router.post("/texto", response_model=PredictionResponse)
def predict_texto(data: TextoInput) -> PredictionResponse:
    prediction = predict_texto_demo(data.texto)
    return PredictionResponse(model_name="texto-demo", prediction=prediction)


@router.post("/imagen", response_model=PredictionResponse)
async def predict_imagen(file: UploadFile = File(...)) -> PredictionResponse:
    content = await file.read()
    prediction = predict_imagen_demo(content)
    return PredictionResponse(model_name="imagen-demo", prediction=prediction)
