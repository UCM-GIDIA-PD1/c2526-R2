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
## 🚀 Guía de Instalación: Ollama y Modelos VLM

Para ejecutar este proyecto de forma local y garantizar la privacidad de los datos, utilizamos **Ollama** como nuestro motor de inferencia. A continuación, se detallan los pasos para configurar el entorno en cualquier sistema operativo.

### 1. Instalación de Ollama

Elige las instrucciones correspondientes a tu sistema operativo:

#### 🪟 Windows
1. Dirígete a la página oficial: [https://ollama.com/download](https://ollama.com/download)
2. Haz clic en **Download for Windows**.
3. Ejecuta el archivo `.exe` descargado y sigue las instrucciones del instalador.
4. Abre el Símbolo del sistema (CMD) o PowerShell para verificar la instalación ejecutando `ollama -v`.

#### 🍏 macOS
1. Dirígete a [https://ollama.com/download](https://ollama.com/download) y descarga la versión para macOS.
2. Descomprime el archivo y arrastra la aplicación **Ollama** a tu carpeta de *Aplicaciones*.
3. Alternativamente, si usas **Homebrew**, puedes instalarlo desde la terminal con:
   ```bash
   brew install ollama
#### 🐧 Instalación en Linux
Abre tu terminal y ejecuta el script de instalación oficial con el siguiente comando:
```bash
curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
```
### 2. Descarga de los Modelos (LLaVA y BakLLaVA)

Una vez que Ollama esté instalado y ejecutándose en segundo plano, abre tu terminal y ejecuta los siguientes comandos para descargar los modelos de Visión-Lenguaje utilizados en este proyecto.

*Nota: La primera vez que ejecutes estos comandos, Ollama descargará los pesos de los modelos (aprox. 4GB a 5GB cada uno). El tiempo dependerá de tu conexión a internet.*

**Descargar el modelo base (LLaVA):**
```bash
ollama run llava
```
*(Para salir del chat interactivo que se abre al terminar la descarga, escribe `/bye` o presiona `Ctrl + D`).*

**Descargar el modelo optimizado (BakLLaVA):**
```bash
ollama run bakllava
```
*(Igualmente, escribe `/bye` para salir una vez descargado).*

---

### 3. Verificación del Entorno

Para confirmar que los modelos se han descargado correctamente y están listos para que el script de Python los utilice, ejecuta el siguiente comando en tu terminal:

```bash
ollama list
```

Deberías ver `llava:latest` y `bakllava:latest` en la lista de modelos disponibles. ¡Tu servidor de inferencia local ya está configurado y listo para clasificar imágenes!


### 6 Evaluacion

#### 💰 Precios
```bash
uv run -m src.6.Evaluacion.1_Precios
```
---
