"""
map_routes.py

Endpoints de la API para servir datos geoespaciales de las rejillas de Madrid.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.services.minio_service import listar_capas, obtener_geojson, REJILLAS_CONFIG

router = APIRouter(prefix="/api/mapas", tags=["mapas"])


@router.get("/capas")
def get_capas():
    """Devuelve el catálogo de rejillas disponibles con sus columnas numéricas."""
    try:
        capas = listar_capas()
        return {"rejillas": capas}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar capas: {e}")


@router.get("/datos")
def get_datos(rejilla: str = Query(..., description="Tipo de rejilla: barrios, secciones_censales, hexagonos_1, hexagonos_2")):
    """Devuelve el GeoJSON completo de una rejilla (todas las columnas)."""
    if rejilla not in REJILLAS_CONFIG:
        raise HTTPException(
            status_code=404,
            detail=f"Rejilla '{rejilla}' no encontrada. Opciones: {list(REJILLAS_CONFIG)}",
        )
    try:
        geojson = obtener_geojson(rejilla)
        return JSONResponse(content=geojson)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener datos: {e}")
