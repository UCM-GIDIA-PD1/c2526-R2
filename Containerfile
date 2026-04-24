FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

# Instalamos únicamente las dependencias estrictamente necesarias para el servidor web.
# Evitamos instalar las dependencias de scraping, análisis y entrenamiento (geopandas, folium, jupyter, etc.)
RUN uv pip install --system \
    "fastapi>=0.116.1" \
    "uvicorn[standard]>=0.35.0" \
    "python-multipart>=0.0.20" \
    "numpy>=2.0.0" \
    "pandas>=3.0.0" \
    "pillow>=12.1.1" \
    "scikit-learn>=1.8.0" \
    "xgboost>=3.2.0" \
    "tensorflow>=2.21.0" \
    "ollama>=0.6.1"

# Copiamos el resto de los archivos.
# El archivo .containerignore filtrará automáticamente todas las carpetas pesadas de datos,
# notebooks, y scripts de entrenamiento para que solo la app y los modelos queden en la imagen.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
