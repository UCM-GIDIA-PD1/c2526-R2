from pydantic import BaseModel, Field


class VentaInput(BaseModel):
    m2: float = Field(..., gt=0)
    habitaciones: int = Field(..., ge=0)
    banos: int = Field(..., ge=0)
    codigo_postal: str


class AlquilerInput(BaseModel):
    m2: float = Field(..., gt=0)
    habitaciones: int = Field(..., ge=0)
    banos: int = Field(..., ge=0)
    codigo_postal: str


class TextoInput(BaseModel):
    texto: str = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    model_name: str
    prediction: str | float | int
