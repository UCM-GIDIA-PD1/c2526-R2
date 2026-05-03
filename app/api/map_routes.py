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

from app.services.minio_service import listar_secundarios, obtener_geojson_secundario

@router.get("/secundarios/capas")
def get_secundarios_capas():
    """Devuelve la lista de datasets secundarios disponibles."""
    try:
        datasets = listar_secundarios()
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar datasets secundarios: {e}")

@router.get("/secundarios/datos")
def get_secundarios_datos(nombre: str = Query(..., description="Nombre del dataset secundario")):
    """Devuelve el GeoJSON de un dataset secundario de puntos."""
    try:
        geojson = obtener_geojson_secundario(nombre)
        return JSONResponse(content=geojson)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener datos secundarios: {e}")
