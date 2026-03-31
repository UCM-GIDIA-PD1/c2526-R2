# 🏠 MAiDay

<img width="478" height="164" alt="Captura_de_pantalla_2026-02-04_205952-removebg-preview" src="https://github.com/user-attachments/assets/70488fc4-ddba-4ee3-a97d-c629fc7c221a" />

---

## 📌 Qué es?

**MAiDay** es el asistente diseñado para equilibrar el sector inmobiliario madrileño. Ofrecemos recomendaciones de precios precisas para propietarios, búsqueda personalizada para inquilinos y evaluación de modernidad mediante IA. Gracias a la inteligencia geográfica, transformamos la incertidumbre en decisiones informadas.

---
## Cómo funciona ?

El sistema articula tres motores tecnológicos complementarios que transforman datos complejos en soluciones directas para el usuario:

**Sistema de Regresión Multivariable 📊**  Un modelo avanzado que conecta todas las variables críticas —como dimensiones, planta y ubicación— para estimar el precio justo de mercado, eliminando los sesgos y las valoraciones arbitrarias.

**Motor Geointeligente de Madrid 🗺️**  Una herramienta visual que cruza datos de fuentes como OSM y el CRTM para representar en un mapa interactivo las ventajas estratégicas de cada vivienda, analizando su conexión con el transporte, servicios y ocio.

**Análisis Visual mediante IA 🧠**  Un sistema de aprendizaje profundo entrenado para examinar las fotografías de los inmuebles y asignar un "Score de Modernidad" objetivo, permitiendo validar el estado de reforma de la vivienda de forma automática.


## 🧠 Arquitectura del sistema

El proyecto se organiza en **5 etapas principales**:

```
src/
│
├── 1_Extraccion     → Obtención de datos de múltiples fuentes
├── 2_Limpieza       → Preprocesamiento y limpieza
├── 3_Agrupacion     → Integración y generación de datasets finales
├── 4_Analisis       → Análisis exploratorio (notebooks)
├── 5_Modelos        → Modelos de ML (precio, texto, imágenes)
```

Cada carpeta contiene su propio `main.py`, que actúa como **pipeline de esa fase**.

---

## ⚙️ Configuración del entorno de desarrollo

Este proyecto utiliza uv como gestor de entornos y dependencias, hay que seguir los siguientes pasos:

### 1. Instalar `uv`

```bash
pip install uv
```
Más información: https://docs.astral.sh/uv/

---

### 2. Crear entorno virtual

```bash
uv venv
```

Activar entorno:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

---

### 3. Instalar dependencias

Como el proyecto contiene pyproject.toml, para instalar las depedencias se puede ejecutar el comando:

```bash
uv sync
```

---

### 4. Verificar versión de Python del entorno

Una vez creado el entorno virtual, se puede comprobar que versión de Python se está usando:

**Windows**
```bash
.\.venv\Scripts\python.exe --version
```

**macOS/Linux**
```bash
./.venv/bin/python --version
```

---

## 🔐 Configuración de MinIO

El proyecto utiliza un servidor MinIO para almacenar y recuperar datasets.

Se debe crear un fichero .env en la raíz del proyecto con el siguiente contenido:

```env
MINIO_ENDPOINT=minio.fdi.ucm.es
MINIO_ACCESS_KEY=TU_ACCESS_KEY
MINIO_SECRET_KEY=TU_SECRET_KEY
MINIO_BUCKET=pd1
MINIO_GROUP_PATH=grupo2
```

### ⚠️ Importante
- NO subir el fichero `.env` a GitHub
- Añadir `.env` al `.gitignore`

## Descripción de las variables

**MINIO_ENDPOINT:** Dirección del servidor MinIO

**MINIO_ACCESS_KEY:** Clave de acceso al servidor

**MINIO_SECRET_KEY:** Clave secreta

**MINIO_BUCKET:** Bucket donde se almacenan los datos

**MINIO_GROUP_PATH:** Carpeta base del grupo dentro del bucket

---

## 🚀 Ejecución del proyecto

Para poder ejecutar los scripts y notebooks, se puede con el siguiente comando en el terminal de visual studio code o por la consola:

```bash
uv run -m (ubicacion del scrip o notebook)
```

Ejemplo:

```bash
uv run -m src.1_Extraccion.A_Anuncios_viviendas
```

---

## 🔄 Pipeline completo de datos

Se recomienda ejecutar las fases en orden:

```
Extracción → Limpieza → Agrupación → Modelos
```

---

### 1 Extracción

```bash
uv run -m src.1_Extraccion.main
```


### 2 Limpieza

```bash
uv run -m src.2_Limpieza.main
```


### 3 Agrupación

```bash
uv run -m src.3_Agrupacion.main
```


### 4 Análisis

```bash
uv run -m src.4_Analisis
```


### 5 Modelos

#### 💰 Precios
```bash
uv run -m src.5_Modelos.1_Precios
```

#### 💬 Texto
```bash
uv run -m src.5_Modelos.2_Texto
```

#### 🖼️ Imágenes
```bash
uv run -m src.5_Modelos.3_Imagenes
```


### 6 Evaluacion

#### 💰 Precios
```bash
uv run -m src.6.Evaluacion.1_Precios
```
---