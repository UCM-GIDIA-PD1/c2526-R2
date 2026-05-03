from typing import List
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.services.image_predictor import predictor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predictions"])

class ImagePredictionResponse(BaseModel):
    filename: str
    clase: str
    probabilidades: dict[str, float]

@router.post("/imagen", response_model=List[ImagePredictionResponse])
async def predict_imagen(files: List[UploadFile] = File(...)):
    """
    Endpoint para procesar y clasificar múltiples imágenes simultáneamente.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos.")

    try:
        image_bytes_list = []
        filenames = []
        
        for file in files:
            # Validar formato
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"El archivo {file.filename} no es una imagen válida.")
            
            content = await file.read()
            image_bytes_list.append(content)
            filenames.append(file.filename)
        
        # Ejecutar inferencia en bloque
        resultados = predictor.predict_batch(image_bytes_list)
        
        response = []
        for filename, res in zip(filenames, resultados):
            response.append(
                ImagePredictionResponse(
                    filename=filename,
                    clase=res["clase"],
                    probabilidades=res["probabilidades"]
                )
            )
            
        return response

    except Exception as e:
        logger.exception("Error en predicción de imágenes")
        raise HTTPException(status_code=500, detail=f"Error interno procesando las imágenes: {e}")
