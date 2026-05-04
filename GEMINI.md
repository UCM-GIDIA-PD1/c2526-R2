# Project Overview: MAiDay

**MAiDay** is an intelligent assistant for the Madrid real estate market, designed to provide data-driven insights for both owners and tenants. It integrates various data sources (public data, scraping, spatial data) to perform three primary tasks:
1.  **Price Estimation:** Recommends sale and rental prices based on market data, socioeconomic variables, and neighborhood characteristics.
2.  **Text Analysis:** Classifies property descriptions to identify the advertiser type (Individual, Intermediary, or Developer).
3.  **Image Classification:** Categorizes property photos into rooms (Bedroom, Kitchen, Living Room, Bathroom).

The project is divided into a **data pipeline** (`src/`) and a **web application** (`app/`) for model serving.

---

## Technical Stack

-   **Language:** Python 3.12+
-   **Package Manager:** [uv](https://docs.astral.sh/uv/)
-   **Web Framework:** FastAPI (API) + Vanilla HTML/JS (Frontend)
-   **Machine Learning:** XGBoost, Scikit-learn, TensorFlow, Keras.
-   **VLM/Vision:** Ollama (LLaVA, BakLLaVA) for image tagging.
-   **Data Storage:** MinIO (Object storage for datasets and artifacts).
-   **Experiment Tracking:** Weights & Biases (WandB).
-   **Spatial Analysis:** GeoPandas, OSMnx, Folium, Shapely, H3.
-   **Containerization:** Podman/Docker.

---

## Directory Structure

-   `app/`: Model serving application.
    -   `api/`: Endpoint definitions (routes).
    -   `core/`: App-specific configuration.
    -   `services/`: Business logic and model predictors.
    -   `web/`: Static frontend files (HTML, CSS, JS).
-   `src/`: Modular data pipeline.
    -   `1_Extraccion/`: Scraping (Idealista) and fetching public data (Ayto Madrid, INE, Catastro, OSM).
    -   `2_Limpieza/`: Data cleaning and preprocessing.
    -   `3_Transformacion/`: Spatial joins, feature engineering, and dataset generation.
    -   `4_Analisis/`: Exploratory Data Analysis (EDA) notebooks.
    -   `5_Modelos/`: Model training scripts for prices, text, and images.
    -   `6_Evaluacion/`: Model evaluation and performance metrics.
    -   `utils/`: Shared utilities (MinIO client, global configuration, WandB setup).
-   `model_cache/`: Local cache for downloaded models.
-   `wandb/`: Local logs and artifacts for Weights & Biases.

---

## Building and Running

### Development Setup
1.  **Install `uv`**: `pip install uv`
2.  **Environment**: `uv venv`
3.  **Dependencies**: `uv sync`

### Running the Pipeline
-   **Interactive Orchestrator**: `uv run -m main` (Executes `src/main.py`)
-   **Specific Phase**: `uv run -m src.1_Extraccion.main` (or any other phase).

### Running the Web App
-   **Local Development**: `uv run uvicorn app.main:app --reload`
    -   Web UI: [http://127.0.0.1:8000](http://127.0.0.1:8000)
    -   Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Containerization
-   **Build**: `podman build -t maiday-web -f Containerfile .`
-   **Run**: `podman run --rm -p 8000:8000 maiday-web`

---

## Development Conventions

-   **Modular Pipeline**: Each phase in `src/` should have its own `main.py` and be independent where possible.
-   **Configuration**:
    -   Use `src/utils/config.py` for global constants and MinIO paths.
    -   Use `.env` for sensitive credentials (MinIO keys).
-   **Data Access**: Use `src/utils/funciones_minio.py` for all interactions with MinIO.
-   **Model Artifacts**: Models should be saved/loaded via `app/services/model_loader.py` for consistent serving.
-   **Logging**: Use WandB for tracking experiments during the `5_Modelos` phase.
-   **VLM Dependency**: Image processing scripts in `src/5_Modelos/3_Imagenes` require a local **Ollama** instance with `llava` or `bakllava` models.

---

## Key Files
-   `pyproject.toml`: Defines all project dependencies.
-   `src/main.py`: Interactive entry point for the entire data pipeline.
-   `app/main.py`: Entry point for the FastAPI serving application.
-   `src/utils/config.py`: Central hub for dataset URLs, MinIO paths, and spatial tags.
-   `.env`: Required for MinIO connection (see `README.md` for template).
