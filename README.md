# MAiDay

<img width="478" height="164" alt="Captura_de_pantalla_2026-02-04_205952-removebg-preview" src="https://github.com/user-attachments/assets/70488fc4-ddba-4ee3-a97d-c629fc7c221a" />


## Qué es?

MAiDay es el asistente diseñado para equilibrar el sector inmobiliario madrileño. Ofrecemos recomendaciones de precios precisas para propietarios, búsqueda personalizada para inquilinos y evaluación de modernidad mediante IA. Gracias a la inteligencia geográfica, transformamos la incertidumbre en decisiones informadas.

---
## Cómo funciona ?

El sistema articula tres motores tecnológicos complementarios que transforman datos complejos en soluciones directas para el usuario:

**Sistema de Regresión Multivariable:** Un modelo avanzado que conecta todas las variables críticas —como dimensiones, planta y ubicación— para estimar el precio justo de mercado, eliminando los sesgos y las valoraciones arbitrarias.

**Motor Geointeligente de Madrid:** Una herramienta visual que cruza datos de fuentes como OSM y el CRTM para representar en un mapa interactivo las ventajas estratégicas de cada vivienda, analizando su conexión con el transporte, servicios y ocio.

**Análisis Visual mediante IA:** Un sistema de aprendizaje profundo entrenado para examinar las fotografías de los inmuebles y asignar un "Score de Modernidad" objetivo, permitiendo validar el estado de reforma de la vivienda de forma automática.

## Configuración del entorno de desarrollo

Este proyecto utiliza uv como gestor de entornos y dependencias, hay que seguir los siguientes pasos:

**Paso 1, Instalar uv**
```bash
pip install uv
```
Más información: https://docs.astral.sh/uv/

**Paso 2, Crear el entorno virtual**
```bash
uv venv
```
para activar el entorno:

**-Windows**
```bash
.venv\Scripts\activate
```
**-macOS/Linux**
```bash
source .venv/bin/activate
```

**Paso 3, Instalar dependencias**

Como el proyecto contiene pyproject.toml, para instalar las depedencias se puede ejecutar el comando:
```bash
uv sync
```
**Paso 4, Verificar la versión de Python del entorno**

Una vez creado el entorno virtual, se puede comprobar que versión de Python se está usando:

**Windows**
```bash
.\.venv\Scripts\python.exe --version
```
**macOS/Linux**
```bash
./.venv/bin/python --version
```

## Uso de MinIO en el proyecto
El proyecto utiliza un servidor MinIO para almacenar y recuperar datasets.

Debes crear un fichero .env en la raíz del proyecto con el siguiente contenido:
```env
MINIO_ENDPOINT=minio.fdi.ucm.es
MINIO_ACCESS_KEY=TU_ACCESS_KEY
MINIO_SECRET_KEY=TU_SECRET_KEY
MINIO_BUCKET=pd1
MINIO_GROUP_PATH=grupo2
```
**Importante:** 

-NO subir el fichero .env a GitHub

-Añadir .env al .gitignore

## Descripción de las variables

**MINIO_ENDPOINT:** Dirección del servidor MinIO

**MINIO_ACCESS_KEY:** Clave de acceso al servidor

**MINIO_SECRET_KEY:** Clave secreta

**MINIO_BUCKET:** Bucket donde se almacenan los datos

**MINIO_GROUP_PATH:** Carpeta base del grupo dentro del bucket