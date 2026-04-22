from fastapi import APIRouter, File, UploadFile

from app.schemas import AlquilerInput, PredictionResponse, TextoInput, VentaInput
from app.services.demo_predictors import (
    predict_imagen_demo,
    predict_texto_demo,
)
from app.services.predictors import predict_tabular


router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post("/venta", response_model=PredictionResponse)
def predict_venta(data: VentaInput) -> PredictionResponse:
    prediction = predict_tabular("venta", data.model_dump())
    return PredictionResponse(model_name="venta-xgboost", prediction=prediction)


@router.post("/alquiler", response_model=PredictionResponse)
def predict_alquiler(data: AlquilerInput) -> PredictionResponse:
    prediction = predict_tabular("alquiler", data.model_dump())
    return PredictionResponse(model_name="alquiler-xgboost", prediction=prediction)


@router.post("/texto", response_model=PredictionResponse)
def predict_texto(data: TextoInput) -> PredictionResponse:
    prediction = predict_texto_demo(data.texto)
    return PredictionResponse(model_name="texto-demo", prediction=prediction)


@router.post("/imagen", response_model=PredictionResponse)
async def predict_imagen(file: UploadFile = File(...)) -> PredictionResponse:
    content = await file.read()
    prediction = predict_imagen_demo(content)
    return PredictionResponse(model_name="imagen-demo", prediction=prediction)
