# ── Etapa única: imagen de producción ──────────────────────────────────────────
FROM python:3.12-slim

# Copiamos los binarios de uv directamente desde la imagen oficial
# (evita instalar pip + uv manualmente y siempre usa la última versión estable)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ── Variables de entorno ────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Permite importaciones cruzadas entre módulos del proyecto (app/, src/, etc.)
    PYTHONPATH="/maiday" \
    # uv: ruta fija para el entorno virtual dentro del contenedor
    UV_PROJECT_ENVIRONMENT="/maiday/.venv"

WORKDIR /maiday

# ── Dependencias de sistema geoespaciales ───────────────────────────────────────
# Este bloque es crítico: geopandas, pyproj y rtree requieren estas librerías nativas.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    proj-bin \
    proj-data \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Caché de dependencias Python ────────────────────────────────────────────────
# Copiamos SOLO los ficheros de metadatos primero para aprovechar el caché de capas:
# si el código fuente cambia pero las dependencias no, esta capa no se reconstruye.
COPY pyproject.toml README.md uv.lock .python-version ./

# Instalamos las dependencias en el entorno virtual sin tocar el código fuente.
# --frozen: usa el lockfile tal cual (reproducibilidad)
# --no-cache: evita almacenar cache de pip dentro de la imagen (imagen más ligera)
RUN uv sync --frozen --no-cache --no-install-project

# Descarga los recursos de NLTK necesarios para el módulo de análisis de texto
RUN /maiday/.venv/bin/python -m nltk.downloader stopwords wordnet punkt_tab

# ── Código fuente ────────────────────────────────────────────────────────────────
# Se copia después de instalar dependencias para maximizar el reuso de caché.
# El archivo .containerignore excluye datos pesados, notebooks y scripts de entrenamiento.
COPY . .

# Segunda sincronización: instala el proyecto ahora que el código fuente está disponible
RUN uv sync --frozen --no-cache

EXPOSE 8000

# Ejecutamos uvicorn desde el entorno virtual creado por uv
CMD ["/maiday/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
