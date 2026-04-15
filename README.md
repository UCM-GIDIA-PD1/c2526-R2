# 🏠 MAiDay

<img width="478" height="164" alt="Captura_de_pantalla_2026-02-04_205952-removebg-preview" src="https://github.com/user-attachments/assets/70488fc4-ddba-4ee3-a97d-c629fc7c221a" />

---

## Qué es?

**MAiDay** es un asistente inteligente diseñado para equilibrar el sector inmobiliario madrileño. El proyecto tiene tres objetivos principales:

1. **Estimación de precios**: Ofrecer recomendaciones de precios, tanto para venta como para alquiler, para propietarios e inquilinos, mediante modelos entrenados con datos reales del mercado y características de la zona.
2. **Análisis de descripciones**: Mediante un modelo de clasificación, según la descripción de los anuncios, se averigua si un anuncio fue publicado por un particular, un intermediario o una promotora.
3. **Análisis de imagenes**: Por medio de un modelo de clasificación se etiquetan imágenes según la habitación que muestren: dormitorio, cocina, salón y baño.

---

## Estructura del repositorio

```
c2526-R2/
│
├── src/                          # Código fuente principal
│   ├── main.py                   # Orquestador principal del pipeline
│   ├── utils/                    # Utilidades compartidas (MinIO, helpers)
│   │
│   ├── 1_Extraccion/             # Fase 1: Obtención de datos de múltiples fuentes
│   │   ├── main.py
│   │   ├── A_Anuncios_viviendas.py   # Scraping de anuncios inmobiliarios
│   │   ├── B_aytoMadrid.py           # Datos del Ayuntamiento de Madrid
│   │   ├── C_transporte.py           # Datos de transporte público (CRTM)
│   │   ├── D_catastro.py             # Datos del Catastro
│   │   ├── E_INE.py                  # Datos del INE
│   │   ├── F_osmnx.py                # Datos OSM (puntos de interés)
│   │   ├── G_rejillas_madrid.py      # Rejilla geográfica de Madrid
│   │   └── H_padron.py               # Datos del Padrón Municipal
│   │
│   ├── 2_Limpieza/               # Fase 2: Preprocesamiento y limpieza de datos
│   │   ├── main.py
│   │   ├── A_viviendas_modificacion.py
│   │   ├── B_aytoMadrid.py
│   │   └── C_transporte.py
│   │
│   ├── 3_Transformacion/         # Fase 3: Integración y generación de datasets finales
│   │   ├── main.py
│   │   ├── A_rejilla_madrid.py
│   │   ├── B_viviendas_madrid.py
│   │   ├── C_aytoMadrid.py
│   │   ├── D_INE_con_geometria.py
│   │   ├── E_Imagenes_div.py
│   │   ├── F_Preparar_datasets_Precios.py
│   │   └── G_texto.py
│   │
│   ├── 4_Analisis/               # Fase 4: Análisis exploratorio (notebooks Jupyter)
│   │   ├── analisis_estadistico.ipynb
│   │   ├── analisis_estadistico_2.ipynb
│   │   ├── analisis_texto.ipynb
│   │   └── analisis_cantidad_imagenes.ipynb
│   │
│   ├── 5_Modelos/                # Fase 5: Entrenamiento de modelos de ML
│   │   ├── main.py
│   │   ├── 1_Precios/            # Modelo de regresión de precios
│   │   ├── 2_Texto/              # Modelo de análisis de texto (NLP)
│   │   └── 3_Imagenes/           # Modelo de análisis de imágenes (VLM)
│   │
│   └── 6_Evaluacion/             # Fase 6: Evaluación de los modelos
│       ├── main.py
│       ├── 1_Precios/            # Métricas del modelo de precios
│       ├── 2_Texto/              # Métricas del modelo de texto
│       └── 3_Imagenes/           # Métricas del modelo de imágenes
│
├── pyproject.toml                # Dependencias y configuración del proyecto
├── uv.lock                       # Lockfile de dependencias
├── .env                          # Variables de entorno (NO subir a Git)
└── .gitignore
```

---

## Configuración del entorno de desarrollo

Este proyecto utiliza [**uv**](https://docs.astral.sh/uv/) como gestor de entornos y dependencias. Requiere **Python ≥ 3.12**.

### 1. Instalar `uv`

```bash
pip install uv
```

### 2. Crear el entorno virtual

```bash
uv venv
```

Activar el entorno:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

### 3. Instalar dependencias

El proyecto define todas sus dependencias en `pyproject.toml`. Para instalarlas ejecuta:

```bash
uv sync
```

Esto instalará automáticamente todos los paquetes necesarios:
`drissionpage`, `folium`, `geopandas`, `geopy`, `googlemaps`, `h3`, `matplotlib`, `minio`, `nltk`, `numpy`, `ollama`, `optuna`, `osmnx`, `pandas`, `pillow`, `plotly`, `pyarrow`, `python-dotenv`, `requests`, `scikit-learn`, `scipy`, `seaborn`, `shapely`, `tensorflow`, `torchvision`, `tqdm`, `umap-learn`, `wandb`, `xgboost`, entre otros.

### 4. Verificar la versión de Python del entorno

**Windows**
```bash
.\.venv\Scripts\python.exe --version
```

**macOS/Linux**
```bash
./.venv/bin/python --version
```

---

## Configuración de variables de entorno (MinIO)

El proyecto utiliza un servidor **MinIO** para almacenar y recuperar datasets. Crea un fichero `.env` en la raíz del proyecto con el siguiente contenido:

```env
MINIO_ENDPOINT=minio.fdi.ucm.es
MINIO_ACCESS_KEY=TU_ACCESS_KEY
MINIO_SECRET_KEY=TU_SECRET_KEY
MINIO_BUCKET=pd1
MINIO_GROUP_PATH=grupo2
```

| Variable | Descripción |
|---|---|
| `MINIO_ENDPOINT` | Dirección del servidor MinIO |
| `MINIO_ACCESS_KEY` | Clave de acceso al servidor |
| `MINIO_SECRET_KEY` | Clave secreta |
| `MINIO_BUCKET` | Bucket donde se almacenan los datos |
| `MINIO_GROUP_PATH` | Carpeta base del grupo dentro del bucket |

---

## Ejecución de los scripts

### Orquestador principal (menú interactivo)

El orquestador detecta automáticamente todas las fases disponibles y permite ejecutarlas de forma interactiva:

```bash
uv run -m main
```

### Ejecución individual de cada fase

Se pueden ejecutar las fases directamente con el comando `uv run -m`:

#### Fase 1 — Extracción
```bash
uv run -m src.1_Extraccion.main
```

#### Fase 2 — Limpieza
```bash
uv run -m src.2_Limpieza.main
```

#### Fase 3 — Transformación
```bash
uv run -m src.3_Transformacion.main
```

#### Fase 4 — Análisis (notebooks Jupyter)

Los notebooks de análisis exploratorio se abren con Jupyter:
```bash
uv run jupyter notebook src/4_Analisis/
```

#### Fase 5 — Modelos

```bash
# Modelo de precios
uv run -m src.5_Modelos.main

# O por submódulo:
uv run -m src.5_Modelos.1_Precios.main
uv run -m src.5_Modelos.2_Texto.main
uv run -m src.5_Modelos.3_Imagenes.main
```

#### Fase 6 — Evaluación

```bash
# Evaluación completa
uv run -m src.6_Evaluacion.main

# O por submódulo:
uv run -m src.6_Evaluacion.1_Precios.main
uv run -m src.6_Evaluacion.2_Texto.main
uv run -m src.6_Evaluacion.3_Imagenes.main
```

### Orden recomendado de ejecución

```
1_Extraccion → 2_Limpieza → 3_Transformacion → 5_Modelos → 6_Evaluacion
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

| GitHub |
|---|
| [@iisma-ai](https://github.com/iisma-ai) |
| [@kauan287](https://github.com/kauan287) |
| [@sperezplaza](https://github.com/sperezplaza) |
| [@arthur-112](https://github.com/arthur-112) |
| [@ouyang157](https://github.com/ouyang157) |
| [@Oscmarin715](https://github.com/Oscmarin715) |

> *Grupo 2 — Repositorio: `c2526-R2`*
