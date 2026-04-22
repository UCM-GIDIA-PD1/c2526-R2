from fastapi import APIRouter
from app.schemas import AlquilerInput, PredictionResponse, TextoInput, VentaInput
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
