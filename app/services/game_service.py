"""
game_service.py

Servicio del minijuego "¿Conoces Madrid mejor que la IA?".

Origen de datos: raw/datos_primarios/venta/
  - Varios .parquet divididos por barrios (batch_<barrio>_n_<N>.parquet)
  - Columnas: id, Barrio, Precio (str), Superficie (str),
              Num_habitaciones, Banyos, Imagenes (list[dict])
  - Imagenes: lista de dicts, cada uno {<habitacion>: bytes}
              Ej: [{"Salón": b"..."}, {"Dormitorio": b"..."}]

Por partida: se elige un .parquet aleatorio (= barrio aleatorio) y
             una vivienda aleatoria de ese archivo. Todo en un solo paso.
"""

import io
import logging
import os
import random
from typing import Any, Optional

import numpy as np
import pandas as pd
import urllib3
from dotenv import load_dotenv
from minio import Minio

logger = logging.getLogger(__name__)

# ── Rutas MinIO ───────────────────────────────────────────────────────────────
RAW_VENTA_PATH = "raw/datos_primarios/venta"

# ── Columna de precio real ─────────────────────────────────────────────────────
COLUMNA_PRECIO = "Precio"

# ── Nombres de campo del struct[5] (del scraper: cont dict keys) ──────────────
# Orden de preferencia para elegir la foto mostrada
_CLASES_IMAGEN = ["Salón", "Dormitorio", "Cocina", "Comedor", "Baño"]

# ── Pistas que se muestran al usuario ─────────────────────────────────────────
COLUMNAS_PISTA = [
    "Superficie", "Num_habitaciones", "Banyos", "Barrio",
]


# ─────────────────────────────────────────────────────────────────────────────
# CLIENTE MINIO
# ─────────────────────────────────────────────────────────────────────────────

def _crear_cliente() -> tuple[Minio, str, str]:
    """Crea cliente MinIO y devuelve (client, bucket, group_path)."""
    load_dotenv()
    endpoint  = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    bucket    = os.getenv("MINIO_BUCKET", "pd1")
    group     = os.getenv("MINIO_GROUP_PATH", "grupo2")

    assert endpoint,   "Falta MINIO_ENDPOINT en .env"
    assert access_key, "Falta MINIO_ACCESS_KEY en .env"
    assert secret_key, "Falta MINIO_SECRET_KEY en .env"

    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=10.0, read=300.0),
        cert_reqs="CERT_NONE",
        retries=urllib3.Retry(
            total=3, backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        ),
    )
    client = Minio(
        endpoint=endpoint, access_key=access_key, secret_key=secret_key,
        secure=True, http_client=http,
    )
    client._region_map[bucket] = "us-east-1"
    return client, bucket, group


def _descargar_parquet(client: Minio, bucket: str, object_name: str) -> pd.DataFrame:
    """Descarga un Parquet desde MinIO y lo devuelve como DataFrame."""
    response = client.get_object(bucket, object_name)
    buffer = io.BytesIO(response.read())
    response.close()
    response.release_conn()
    return pd.read_parquet(buffer)


# ─────────────────────────────────────────────────────────────────────────────
# SANITIZACIÓN (features → dict JSON-safe)
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_value(v: Any) -> Any:
    """Convierte un valor a tipo Python nativo serializable por JSON/Pydantic."""
    if isinstance(v, (bytes, bytearray)):
        return None
    if isinstance(v, (list, tuple)):
        clean = [_sanitize_value(i) for i in v]
        return None if all(i is None for i in clean) else clean
    if isinstance(v, dict):
        # Descartar dicts cuyo único valor sea bytes (structs de imagen)
        clean = {k: _sanitize_value(val) for k, val in v.items()}
        clean = {k: val for k, val in clean.items() if val is not None}
        return clean if clean else None
    if isinstance(v, float) and np.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, np.integer):  return int(v)
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.bool_):    return bool(v)
    if isinstance(v, np.ndarray):  return v.tolist()
    if isinstance(v, (int, float, str, bool)):
        return v
    return None


def _sanitize_row(row: pd.Series, exclude: list[str]) -> dict[str, Any]:
    """Convierte una fila en dict limpio, excluyendo las columnas indicadas."""
    result: dict[str, Any] = {}
    for col, val in row.items():
        if col in exclude:
            continue
        clean = _sanitize_value(val)
        if clean is not None:
            result[col] = clean
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS  (barrio aleatorio → vivienda aleatoria)
# ─────────────────────────────────────────────────────────────────────────────

def cargar_vivienda_aleatoria() -> tuple[dict[str, Any], float, Optional[bytes]]:
    """
    Selecciona un .parquet aleatorio de raw/datos_primarios/venta/ (un barrio),
    elige una vivienda aleatoria con precio válido y devuelve sus datos.

    Returns:
        features   (dict)       : Features sanitizadas para el modelo.
        precio_real (float)     : Precio real de la vivienda.
        imagen_bytes (bytes|None): Bytes de la primera imagen disponible.

    Raises:
        ValueError: Si no hay parquets o ninguna vivienda tiene precio válido.
    """
    client, bucket, group = _crear_cliente()
    prefix = f"{group}/{RAW_VENTA_PATH}/"

    # 1. Listar todos los .parquet disponibles
    objetos = list(client.list_objects(bucket, prefix=prefix, recursive=True))
    parquets = [
        obj.object_name for obj in objetos
        if obj.object_name.endswith(".parquet")
    ]

    if not parquets:
        raise ValueError(f"No hay archivos .parquet en '{prefix}'.")

    # 2. Elegir un parquet (barrio) aleatorio
    parquet_elegido = random.choice(parquets)
    barrio_nombre = parquet_elegido.split("/")[-1].replace(".parquet", "")
    logger.info("Barrio seleccionado: '%s'", barrio_nombre)

    df = _descargar_parquet(client, bucket, parquet_elegido)

    # 3. Convertir Precio y Superficie a numérico (vienen como str en el raw)
    df[COLUMNA_PRECIO] = (
        df[COLUMNA_PRECIO]
        .astype(str)
        .str.replace(r"[^\d,\.]", "", regex=True)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    if "Superficie" in df.columns:
        df["Superficie"] = (
            df["Superficie"]
            .astype(str)
            .str.replace(r"[^\d,\.]", "", regex=True)
            .str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )

    # 4. Filtrar viviendas con precio válido
    df_valido = df[df[COLUMNA_PRECIO].notna() & (df[COLUMNA_PRECIO] > 0)]
    if df_valido.empty:
        raise ValueError(
            f"No hay viviendas con precio válido en '{barrio_nombre}'."
        )

    # 5. Elegir una vivienda aleatoria
    fila = df_valido.sample(1).iloc[0]
    precio_real = float(fila[COLUMNA_PRECIO])

    # 6. Extraer imagen directamente de la columna Imagenes
    imagen_bytes = obtener_imagen_de_fila(fila)

    # 7. Sanitizar features (excluir columnas no serializables)
    features = _sanitize_row(fila, exclude=[COLUMNA_PRECIO, "Imagenes"])

    logger.info(
        "Vivienda seleccionada — barrio: %s, precio: %.0f €, imagen: %s",
        barrio_nombre, precio_real,
        f"{len(imagen_bytes)} bytes" if imagen_bytes else "no disponible",
    )

    return features, precio_real, imagen_bytes


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTRACCIÓN DE IMAGEN (directo de la fila)
# ─────────────────────────────────────────────────────────────────────────────

def _a_bytes(valor: Any) -> Optional[bytes]:
    """
    Convierte CUALQUIER tipo de valor binario a bytes Python puros.
    Cubre: bytes, bytearray, memoryview, numpy.bytes_, numpy.ndarray binario.
    Devuelve None si el valor es nulo o está vacío.
    """
    if valor is None:
        return None
    # Nulos de NumPy / pandas (NaN, pd.NA, etc.)
    try:
        if pd.isna(valor):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(valor, bytes):
        return valor or None
    if isinstance(valor, bytearray):
        return bytes(valor) or None
    if isinstance(valor, memoryview):
        b = bytes(valor)
        return b or None
    # numpy bytes scalar (dtype='S...' o 'V...')
    if isinstance(valor, (np.bytes_, np.void)):
        try:
            b = bytes(valor)
            return b or None
        except Exception:
            pass
    # numpy ndarray binario (poco frecuente pero posible)
    if isinstance(valor, np.ndarray):
        if valor.dtype.kind in ("S", "V", "u", "i"):
            b = valor.tobytes()
            return b or None
    # Último recurso: intentar conversión genérica
    try:
        b = bytes(valor)
        return b or None
    except Exception:
        return None


def _normalizar_imagenes(raw: Any) -> list[dict]:
    """
    Normaliza el valor de la columna 'Imagenes' a una lista Python de dicts.

    Pandas puede devolver este campo como:
      - list[dict]            → el caso ideal
      - numpy.ndarray         → lista empaquetada como array
      - list[numpy.void]      → elementos de struct como numpy.void
      - list[numpy.ndarray]   → structs como arrays estructurados

    Returns:
        Lista de dicts {clase: bytes|None}. Vacía si no hay datos válidos.
    """
    if raw is None:
        return []

    # 1. Si es ndarray, convertir a lista Python
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()

    # 2. Asegurar que es iterable de algún tipo
    if not isinstance(raw, (list, tuple)):
        logger.debug("Imagenes: tipo inesperado %s, intentando list()", type(raw).__name__)
        try:
            raw = list(raw)
        except Exception:
            return []

    resultado: list[dict] = []
    for item in raw:
        if isinstance(item, dict):
            resultado.append(item)
        elif isinstance(item, np.void):
            # numpy.void: elemento de un struct de PyArrow → convertir a dict
            d = {name: item[name] for name in item.dtype.names}
            resultado.append(d)
        elif isinstance(item, np.ndarray) and item.dtype.names:
            # numpy structured array
            d = {name: item[name].item() for name in item.dtype.names}
            resultado.append(d)
        elif item is not None:
            logger.debug("Imagenes: elemento de tipo %s ignorado", type(item).__name__)

    return resultado


def obtener_imagen_de_fila(fila: pd.Series) -> Optional[bytes]:
    """
    Extrae los bytes WebP de la primera imagen disponible de la columna 'Imagenes'.

    Estructura del scraper: list[dict] donde cada dict tiene UNA key (la habitación)
    y el valor son bytes WebP.
        Ejemplo: [{"Salón": b"RIFF...WEBP"}, {"Dormitorio": b"RIFF...WEBP"}]

    Estructura en Parquet (PyArrow list[struct[5]]):
        Cada elemento puede ser un dict con TODOS los campos del struct,
        donde solo uno tiene bytes y el resto son None.
        Ejemplo: [{"Salón": b"...", "Dormitorio": None, "Cocina": None, ...}]

    Returns:
        bytes WebP, o None si no hay imágenes.
    """
    raw = fila["Imagenes"] if "Imagenes" in fila.index else None
    imagenes = _normalizar_imagenes(raw)

    if not imagenes:
        logger.debug("Imagenes: lista vacía tras normalización (tipo original: %s).", type(raw).__name__)
        return None

    logger.debug("Imagenes: %d elementos encontrados.", len(imagenes))

    # Recopilar por clase (cada dict puede tener 1 o 5 keys)
    por_clase: dict[str, bytes] = {}
    primera_disponible: Optional[bytes] = None

    for item in imagenes:
        for clase, valor in item.items():
            img_bytes = _a_bytes(valor)
            if img_bytes and clase not in por_clase:
                por_clase[clase] = img_bytes
                if primera_disponible is None:
                    primera_disponible = img_bytes

    if not por_clase:
        logger.debug("Imagenes: ningún campo con bytes válidos. Keys encontradas: %s",
                     [list(i.keys()) for i in imagenes[:3]])
        return None

    # Devolver en orden de preferencia
    for clase in _CLASES_IMAGEN:
        if clase in por_clase:
            logger.info("Imagen: clase='%s', %d bytes.", clase, len(por_clase[clase]))
            return por_clase[clase]

    # Fallback: primera encontrada
    logger.info("Imagen fallback: %d bytes.", len(primera_disponible))
    return primera_disponible


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREDICCIÓN DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────

def predecir_modelo(features: dict[str, Any]) -> float:
    """
    Llama a predict_tabular("venta", features) y multiplica por superficie.

    Returns:
        Precio total estimado en euros.

    Raises:
        RuntimeError: Si el modelo no puede predecir.
    """
    from app.services.predictors import predict_tabular

    try:
        prediction_m2 = predict_tabular("venta", features)
        superficie = float(features.get("Superficie") or 1.0)
        precio_total = float(prediction_m2) * superficie
        logger.info(
            "Predicción modelo — %.2f €/m² × %.1f m² = %.0f €",
            prediction_m2, superficie, precio_total,
        )
        return precio_total
    except Exception as e:
        logger.error("Error al predecir con el modelo: %s", e)
        raise RuntimeError(f"El modelo no pudo generar una predicción: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# 4. CÁLCULO DEL GANADOR
# ─────────────────────────────────────────────────────────────────────────────

def calcular_ganador(
    precio_usuario: float,
    precio_modelo: float,
    precio_real: float,
) -> dict[str, Any]:
    """
    Compara errores del usuario y del modelo respecto al precio real.

    Returns:
        dict con errores absolutos/relativos, ganador y mensaje.
    """
    error_usuario = abs(precio_usuario - precio_real)
    error_modelo  = abs(precio_modelo  - precio_real)
    pct_usuario   = (error_usuario / precio_real) * 100
    pct_modelo    = (error_modelo  / precio_real) * 100

    if error_usuario < error_modelo:
        ganador     = "usuario"
        diferencia  = error_modelo - error_usuario
        mensaje     = (
            f"¡Enhorabuena! Te acercaste {diferencia:,.0f} € más que el modelo. "
            "¡Tu intuición supera a la IA!"
        )
    elif error_modelo < error_usuario:
        ganador     = "modelo"
        diferencia  = error_usuario - error_modelo
        mensaje     = (
            f"El modelo acertó {diferencia:,.0f} € más que tú. "
            "¡La IA ha ganado esta vez!"
        )
    else:
        ganador     = "empate"
        diferencia  = 0.0
        mensaje     = "¡Empate perfecto! Tienes el instinto de una IA."

    return {
        "precio_real":        round(precio_real,    2),
        "precio_usuario":     round(precio_usuario,  2),
        "precio_modelo":      round(precio_modelo,   2),
        "error_usuario":      round(error_usuario,   2),
        "error_modelo":       round(error_modelo,    2),
        "pct_error_usuario":  round(pct_usuario,     1),
        "pct_error_modelo":   round(pct_modelo,      1),
        "ganador":            ganador,
        "diferencia":         round(diferencia,      2),
        "mensaje":            mensaje,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. PISTAS LEGIBLES
# ─────────────────────────────────────────────────────────────────────────────

def extraer_pistas(features: dict[str, Any]) -> dict[str, Any]:
    """
    Formatea las columnas de COLUMNAS_PISTA para mostrar al usuario.
    """
    pistas: dict[str, Any] = {}
    for col in COLUMNAS_PISTA:
        valor = features.get(col)
        if valor is None:
            continue
        try:
            if col == "Superficie":
                pistas[col] = f"{float(valor):.0f} m²"
            elif col in ("Num_habitaciones", "Banyos"):
                pistas[col] = int(float(valor))
            else:
                pistas[col] = valor
        except (TypeError, ValueError):
            pistas[col] = valor
    return pistas
