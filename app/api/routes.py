import logging

from fastapi import APIRouter, HTTPException
from app.schemas import (
    AlquilerInput, AlquilerSimpleInput,
    EnrichedPredictionResponse, PredictionResponse,
    TextoInput, VentaInput, VentaSimpleInput,
)
from app.services.predictors import predict_tabular
from app.services.enrichment import enrich_property

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predictions"])


# ─── Simplified endpoints (auto-enrichment) ────────────────────────────

@router.post("/venta/simple", response_model=EnrichedPredictionResponse)
def predict_venta_simple(data: VentaSimpleInput) -> EnrichedPredictionResponse:
    """
    Predicción de venta simplificada.
    Solo requiere dirección y datos básicos del inmueble.
    Las distancias, transporte, demografía etc. se calculan automáticamente.
    """
    try:
        basic = data.model_dump(exclude={"Direccion"})
        enriched = enrich_property(data.Direccion, basic)
        prediction_m2 = predict_tabular("venta", enriched)
        prediction_total = prediction_m2 * enriched.get("Superficie", 1.0)
        return EnrichedPredictionResponse(
            model_name="venta-xgboost",
            prediction=prediction_total,
            prediction_m2=prediction_m2,
            lat=enriched.get("lat"),
            lon=enriched.get("lon"),
            features_computed={
                k: v for k, v in enriched.items()
                if k not in basic and k not in ("lat", "lon")
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error en predicción de venta simplificada")
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")


@router.post("/alquiler/simple", response_model=EnrichedPredictionResponse)
def predict_alquiler_simple(data: AlquilerSimpleInput) -> EnrichedPredictionResponse:
    """
    Predicción de alquiler simplificada.
    Solo requiere dirección y datos básicos del inmueble.
    """
    try:
        basic = data.model_dump(exclude={"Direccion"})
        enriched = enrich_property(data.Direccion, basic)
        prediction = predict_tabular("alquiler", enriched)
        return EnrichedPredictionResponse(
            model_name="alquiler-xgboost",
            prediction=prediction,
            prediction_m2=prediction / enriched.get("Superficie", 1.0),
            lat=enriched.get("lat"),
            lon=enriched.get("lon"),
            features_computed={
                k: v for k, v in enriched.items()
                if k not in basic and k not in ("lat", "lon")
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error en predicción de alquiler simplificada")
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")


# ─── Original full endpoints (backward compatible) ─────────────────────

@router.post("/venta", response_model=PredictionResponse)
def predict_venta(data: VentaInput) -> PredictionResponse:
    prediction_m2 = predict_tabular("venta", data.model_dump())
    prediction_total = prediction_m2 * data.Superficie
    return PredictionResponse(model_name="venta-xgboost", prediction=prediction_total)

@router.post("/alquiler", response_model=PredictionResponse)
def predict_alquiler(data: AlquilerInput) -> PredictionResponse:
    prediction = predict_tabular("alquiler", data.model_dump())
    return PredictionResponse(model_name="alquiler-xgboost", prediction=prediction)
