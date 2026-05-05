# MAiDay

<div align="center">
  <img width="478" height="164" alt="MAiDay Logo" src="https://github.com/user-attachments/assets/70488fc4-ddba-4ee3-a97d-c629fc7c221a" />
</div>

> **MAiDay** es un asistente inteligente diseñado para equilibrar el mercado inmobiliario madrileño. Integrando fuentes de datos diversas, proporciona información valiosa tanto a propietarios como a inquilinos.

---

## Objetivos del Proyecto

Nuestro asistente cuenta con tres objetivos fundamentales:

1. **Estimación de Precios**: Recomendaciones de precios (venta y alquiler) basadas en datos reales del mercado, variables socioeconómicas y características de la zona.
2. **Análisis de Texto**: Clasificación de descripciones de anuncios para identificar el tipo de anunciante (particular, intermediario o promotora).
3. **Clasificación de Imágenes**: Etiquetado de fotografías inmobiliarias para categorizar las estancias mostradas (dormitorio, cocina, salón y baño).

---

## Estructura del Repositorio

El proyecto sigue un diseño modular y estructurado basado en fases (del 1 al 6) dentro de la carpeta `src/`. Además, cuenta con una aplicación web en `app/`.

```text
c2526-R2/
├── app/                          # Aplicación Web (FastAPI + Frontend)
│   ├── api/                      # Endpoints de la API
│   ├── core/                     # Configuración
│   ├── services/                 # Servicios de inferencia (predictores)
│   ├── web/                      # Archivos estáticos (HTML, CSS, JS)
│   └── main.py                   # Entrypoint de FastAPI
│
├── src/                          # Código fuente principal (Pipeline)
│   ├── main.py                   # Orquestador principal del pipeline
│   ├── utils/                    # Utilidades compartidas
│   ├── model_artifacts/          # Artefactos y modelos guardados
│   │
│   ├── 1_Extraccion/             # Fase 1: Scraping y obtención de datos públicos
│   ├── 2_Limpieza/               # Fase 2: Limpieza y preprocesamiento de datos
│   ├── 3_Transformacion/         # Fase 3: Integración, cruce espacial y generación de datasets
│   ├── 4_Analisis/               # Fase 4: Análisis exploratorio (Notebooks)
│   ├── 5_Modelos/                # Fase 5: Entrenamiento de modelos (Precios, Texto, Imágenes)
│   └── 6_Evaluacion/             # Fase 6: Evaluación y métricas de los modelos
│
├── .env                          # Variables de entorno (MinIO, WandB)
├── Containerfile                 # Configuración de Docker/Podman
├── pyproject.toml                # Dependencias (uv)
├── uv.lock                       # Lockfile
└── README.md                     # Documentación principal
```
---

## Configuración del entorno de desarrollo

Este proyecto utiliza [**uv**](https://docs.astral.sh/uv/) como gestor de entornos y dependencias. Requiere **Python ≥ 3.12**.

### 1. Instalar uv

```bash
pip install uv
```

### 2. Crear y activar el entorno virtual

Genera el entorno virtual:
```bash
uv venv
```

Actívalo según tu sistema operativo:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Instalar dependencias

Sincroniza el entorno con el archivo `pyproject.toml` (instala todas las librerías necesarias, incluyendo TensorFlow, XGBoost, FastAPI, etc.):
```bash
uv sync
```

---

## Ejecución del Pipeline (Scripts)

> Todos los comandos deben ejecutarse desde la raíz del proyecto (`c2526-R2/`) con el entorno virtual activado.

### Orquestador Principal

El orquestador interactivo detecta y permite ejecutar cualquier fase cómodamente.

```bash
uv run -m main
```

### Ejecución de una fase o script específico

Si prefieres no usar el orquestador y quieres ejecutar el `main.py` de una fase concreta cualquiera (por ejemplo, la Fase 1 de Extracción o la Fase 6 de Evaluación), puedes llamarlo directamente indicando su ruta:

```bash
# Ejemplo: Ejecutar el main de la fase de Extracción
uv run -m src.1_Extraccion.main

# Ejemplo: Ejecutar el main de la evaluación del modelo de precios
uv run -m src.6_Evaluacion.1_Precios.main
```
---

## Resultados de Modelos

A continuación se resumen los resultados principales obtenidos en nuestros modelos predictivos y de clasificación:

| Mejor modelo | Tarea | Métrica Principal | Baseline | Resultado |
| :--- | :--- | :--- | :--- | :--- |
| *xgboost* | Estimación precio venta | MAPE | 40.8% | 15.03% |
| *xgboost* | Estimación precio alquiler | MAPE | 67.02% | 15.69% |
| *SVM* | Clasificador de anunciante | F1-score | 0.29 | 0.89 |
| *MLP* | Clasificador de imágenes | F1-score | 0.25 | 0.90 |
---

## Aplicación Web y Contenedores

MAiDay incluye una aplicación web demostrativa para servir los modelos mediante FastAPI.

### Ejecutar la aplicación en local

Con el entorno virtual activado, lanza el servidor web:

```bash
uv run uvicorn app.main:app --reload
```
- **Aplicación Web**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Documentación API (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Construcción y Despliegue con Podman

Para un despliegue aislado, puedes construir el contenedor utilizando el archivo `Containerfile`.

**Requisito previo:** Debes tener instalado [Podman Desktop](https://podman.io/).

> *Nota para usuarios de Podman en Windows:* Recuerda iniciar la máquina virtual antes de ejecutar los comandos:
> ```bash
> podman machine start
> ```

```bash
# Construir la imagen
podman build -t maiday-app -f Containerfile .

# Ejecutar el contenedor (comando completo de producción)
podman run --rm -p 8000:8000 \
    --env-file .env \
    --add-host host.containers.internal:host-gateway \
    maiday-app
```

> **¿Por qué `--add-host host.containers.internal:host-gateway`?**
> Dentro del contenedor, `localhost` apunta al propio contenedor, no a tu máquina. Este flag crea un alias DNS que resuelve `host.containers.internal` hacia la IP de tu host, permitiendo que el contenedor acceda a MinIO u otros servicios corriendo en local. Si `MINIO_ENDPOINT` apunta a `localhost` o `127.0.0.1`, la aplicación lo detecta automáticamente y redirige la conexión a `host.containers.internal`.
---

## Configuraciones Especiales

### Conexión a MinIO (Almacenamiento de Datos)

El proyecto lee y escribe datasets en un servidor MinIO, y utiliza Weights & Biases para el seguimiento de experimentos. Debes crear un archivo `.env` en la raíz del proyecto con las siguientes credenciales:

| Variable | Descripción |
|---|---|
| `MINIO_ENDPOINT` | Dirección del servidor MinIO |
| `MINIO_ACCESS_KEY` | Clave de acceso al servidor |
| `MINIO_SECRET_KEY` | Clave secreta |
| `MINIO_BUCKET` | Bucket donde se almacenan los datos |
| `MINIO_GROUP_PATH` | Carpeta base del grupo dentro del bucket |
| `WANDB_API_KEY` | API Key de Weights & Biases (seguimiento de experimentos) |

```env
#Las claves no deben estar entre comillas para que no de problemas en podman
MINIO_ENDPOINT=minio.fdi.ucm.es
MINIO_ACCESS_KEY=TU_ACCESS_KEY
MINIO_SECRET_KEY=TU_SECRET_KEY
MINIO_BUCKET=pd1
MINIO_GROUP_PATH=grupo2
WANDB_API_KEY=TU_WANDB_API_KEY
```
---
## Configuración de Ollama (modelos VLM)

La fase de análisis de imágenes requiere **Ollama** ejecutándose localmente.

### Instalación de Ollama

| SO | Instrucciones |
|---|---|
| **Windows** | Descarga el instalador en [ollama.com/download](https://ollama.com/download) y ejecuta el `.exe` |
| **macOS** | Descarga en [ollama.com/download](https://ollama.com/download) o usa `brew install ollama` |
| **Linux** | `curl -fsSL https://ollama.com/install.sh \| sh` |

### Descarga de modelos

```bash
# Modelo base (LLaVA) — aprox. 4-5 GB
ollama run llava

# Modelo optimizado (BakLLaVA) — aprox. 4-5 GB
ollama run bakllava
```

> Escribe `/bye` o presiona `Ctrl+D` para salir del chat interactivo tras la descarga.

### Verificación

```bash
ollama list
# Deberías ver: llava:latest  bakllava:latest
```

---

## Equipo de desarrollo

Proyecto desarrollado para la asignatura **Proyecto de Datos I (PD1)** — Universidad Complutense de Madrid (UCM), Grado en Ingeniería de Datos e Inteligencia Artificial.

**Grupo 2**

| Desarrollador | GitHub |
| :--- | :--- |
| iisma-ai | [@iisma-ai](https://github.com/iisma-ai) |
| kauan287 | [@kauan287](https://github.com/kauan287) |
| sperezplaza | [@sperezplaza](https://github.com/sperezplaza) |
| arthur-112 | [@arthur-112](https://github.com/arthur-112) |
| ouyang157 | [@ouyang157](https://github.com/ouyang157) |
| Oscmarin715 | [@Oscmarin715](https://github.com/Oscmarin715) |
