FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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

RUN pip install --no-cache-dir uv

# Instalamos las dependencias necesarias para el servidor web y el servicio de enriquecimiento.
RUN uv pip install --system \
    "fastapi>=0.116.1" \
    "uvicorn[standard]>=0.35.0" \
    "python-multipart>=0.0.20" \
    "numpy>=2.0.0" \
    "pandas>=3.0.0" \
    "pillow>=12.1.1" \
    "scikit-learn>=1.8.0" \
    "xgboost>=3.2.0" \
    "tensorflow-cpu>=2.21.0" \
    "ollama>=0.6.1" \
    "geopandas>=1.1.2" \
    "geopy>=2.4.1" \
    "minio>=7.2.20" \
    "python-dotenv>=1.0.0" \
    "scipy>=1.17.1" \
    "urllib3>=2.0.0" \
    "wandb>=0.15.0" \
    "pyarrow>=15.0.0"

# Copiamos el resto de los archivos.
# El archivo .containerignore filtrará automáticamente todas las carpetas pesadas de datos,
# notebooks, y scripts de entrenamiento para que solo la app y los modelos queden en la imagen.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
