from pydantic import BaseModel
from typing import Any, Optional


# ─── Simplified input schemas (user only provides what can't be derived) ───

class VentaSimpleInput(BaseModel):
    """Datos mínimos que el usuario introduce para predecir venta."""
    Direccion: str                   # Dirección exacta → geocodificación automática
    Distrito: str                    # Distrito de Madrid
    Superficie: float                # m²
    Num_habitaciones: float
    Banyos: float
    Planta: float
    Ventanas: str                    # Tipo de ventanas (Climalit, etc.)
    Ascensor: float                  # 0 o 1
    Terraza: float                   # 0 o 1
    Balcon: float                    # 0 o 1
    Orientacion: str                 # Sur, Norte, Este, Oeste...
    Consumo: str                     # Certificado energético (A-G)
    Anuncia: str                     # Particular / Agencia
    anio_construccion: Optional[float] = None  # Opcional, se calcula del catastro si no se da


class AlquilerSimpleInput(BaseModel):
    """Datos mínimos que el usuario introduce para predecir alquiler."""
    Direccion: str
    Distrito: str
    Superficie: float
    Num_habitaciones: float
    Banyos: float
    Planta: float
    Ventanas: str
    Ascensor: float
    Terraza: float
    Balcon: float
    Equipamiento: float             # 0 o 1 (amueblado)
    Orientacion: str
    Consumo: str
    Anuncia: str
    anio_construccion: Optional[float] = None


# ─── Full input schemas (kept for backward compatibility / direct API) ───

class VentaInput(BaseModel):
    Distrito: str
    Superficie: float
    Num_habitaciones: float
    Banyos: float
    Planta: float
    Ventanas: str
    Ascensor: float
    Terraza: float
    Balcon: float
    Orientacion: str
    Consumo: str
    Anuncia: str
    lat: float
    lon: float
    dist_min_alimentacion: float
    dist_min_bibliotecas: float
    cantidad_bibliotecas_cerca: float
    dist_min_bomberos: float
    cantidad_bomberos_cerca: float
    dist_min_cementerios: float
    cantidad_cementerios_cerca: float
    dist_min_centros_dia: float
    cantidad_centros_dia_cerca: float
    dist_min_centros_educativos: float
    cantidad_centros_educativos_cerca: float
    dist_min_centros_mayores: float
    cantidad_centros_mayores_cerca: float
    dist_min_centros_sociales: float
    cantidad_centros_sociales_cerca: float
    dist_min_comercios: float
    cantidad_comercios_cerca: float
    dist_min_comisarias: float
    cantidad_comisarias_cerca: float
    dist_min_hospitales: float
    cantidad_hospitales_cerca: float
    dist_min_iglesias: float
    cantidad_iglesias_cerca: float
    dist_min_negativos: float
    cantidad_negativos_cerca: float
    dist_min_parques: float
    cantidad_parques_cerca: float
    dist_min_parques_bomberos: float
    cantidad_parques_bomberos_cerca: float
    dist_min_polideportivos: float
    cantidad_polideportivos_cerca: float
    dist_min_puntos_limpios: float
    cantidad_puntos_limpios_cerca: float
    dist_min_servicios_sociales: float
    cantidad_servicios_sociales_cerca: float
    dist_min_universidades: float
    cantidad_universidades_cerca: float
    dist_min_paradas: float
    paradas_cerca: float
    lineas_distintas_paradas_cerca: float
    dist_min_estaciones: float
    lineas_distintas_estaciones_cerca: float
    anio_construccion: float
    poblacion_total: float
    pct_extranjeros: float
    pct_mayores_65: float
    pct_jovenes_30: float

class AlquilerInput(BaseModel):
    Distrito: str
    Superficie: float
    Num_habitaciones: float
    Banyos: float
    Planta: float
    Ventanas: str
    Ascensor: float
    Terraza: float
    Balcon: float
    Equipamiento: float
    Orientacion: str
    Consumo: str
    Anuncia: str
    lat: float
    lon: float
    dist_min_alimentacion: float
    dist_min_bibliotecas: float
    cantidad_bibliotecas_cerca: float
    dist_min_bomberos: float
    cantidad_bomberos_cerca: float
    dist_min_cementerios: float
    cantidad_cementerios_cerca: float
    dist_min_centros_dia: float
    cantidad_centros_dia_cerca: float
    dist_min_centros_educativos: float
    cantidad_centros_educativos_cerca: float
    dist_min_centros_mayores: float
    cantidad_centros_mayores_cerca: float
    dist_min_centros_sociales: float
    cantidad_centros_sociales_cerca: float
    dist_min_comercios: float
    cantidad_comercios_cerca: float
    dist_min_comisarias: float
    cantidad_comisarias_cerca: float
    dist_min_hospitales: float
    cantidad_hospitales_cerca: float
    dist_min_iglesias: float
    cantidad_iglesias_cerca: float
    dist_min_negativos: float
    cantidad_negativos_cerca: float
    dist_min_parques: float
    cantidad_parques_cerca: float
    dist_min_parques_bomberos: float
    cantidad_parques_bomberos_cerca: float
    dist_min_polideportivos: float
    cantidad_polideportivos_cerca: float
    dist_min_puntos_limpios: float
    cantidad_puntos_limpios_cerca: float
    dist_min_servicios_sociales: float
    cantidad_servicios_sociales_cerca: float
    dist_min_universidades: float
    cantidad_universidades_cerca: float
    dist_min_paradas: float
    paradas_cerca: float
    lineas_distintas_paradas_cerca: float
    dist_min_estaciones: float
    lineas_distintas_estaciones_cerca: float
    anio_construccion: float
    poblacion_total: float
    pct_extranjeros: float
    pct_mayores_65: float
    pct_jovenes_30: float

class TextoInput(BaseModel):
    texto: str

class PredictionResponse(BaseModel):
    model_name: str
    prediction: float | str | int

class EnrichedPredictionResponse(BaseModel):
    model_name: str
    prediction: float | str | int
    prediction_m2: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    features_computed: Optional[dict] = None
