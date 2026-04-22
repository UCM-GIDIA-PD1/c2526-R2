from pydantic import BaseModel, Field


class VentaInput(BaseModel):
    model_config = {"extra": "allow"}

class AlquilerInput(BaseModel):
    model_config = {"extra": "allow"}


class TextoInput(BaseModel):
    texto: str = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    model_name: str
    prediction: str | float | int
