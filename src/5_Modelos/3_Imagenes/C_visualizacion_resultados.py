# -*- coding: utf-8 -*-
"""C_visualizacion_resultados.py — Visualización tipo Fashion-MNIST"""

import io
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

from utils.funciones_minio import crear_cliente_minio, bajar_minio, buscar_todos_los_archivos
from utils.config import CLASES_IMAGENES, CNN_TARGET_SIZE

# Máximo de Parquets por clase para visualización
MAX_PARQUETS = 5


# ── Descargar imágenes ───────────────────────────────────────────────

def descargar_imagenes(clases, max_parquets=MAX_PARQUETS):
    """Baja los primeros N Parquets por clase y devuelve [(bytes, clase_id), ...]"""
    cliente = crear_cliente_minio()
    pool = []

    for clase_id, clase in enumerate(clases):
        archivos = buscar_todos_los_archivos(cliente, f"cleaned/dataset_vision/{clase}")
        archivos = archivos[:max_parquets]

        for archivo in archivos:
            df = bajar_minio(cliente, f"cleaned/dataset_vision/{clase}", archivo)
            for raw in df['imagen_bytes']:
                pool.append((bytes(raw), clase_id))
            del df

        print(f"  {clase}: cargado")

    print(f"  Total: {len(pool):,} imagenes")
    return pool


def decodificar(jpeg_bytes):
    """Decodifica JPEG desde bytes, redimensiona y normaliza a [0,1]."""
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    img = img.resize((CNN_TARGET_SIZE[1], CNN_TARGET_SIZE[0]))
    return np.array(img, dtype=np.float32) / 255.0


# ── Cuadrícula del pool (estilo Fashion-MNIST) ───────────────────────

def graficar_pool(pool, clases, filas=5, columnas=5):
    """Cuadrícula 5x5 con imágenes aleatorias y nombre de clase."""
    indices = random.sample(range(len(pool)), filas * columnas)

    plt.figure(figsize=(2 * columnas, 2 * filas))
    for i, idx in enumerate(indices):
        jpeg_bytes, clase_id = pool[idx]
        img = decodificar(jpeg_bytes)

        plt.subplot(filas, columnas, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img)
        plt.xlabel(clases[clase_id])

    plt.suptitle("Pool de Habitaciones", fontsize=14)
    plt.tight_layout()
    plt.savefig('vis_pool_habitaciones.png', dpi=150)
    plt.show()
    print("  Guardado: vis_pool_habitaciones.png")


# ── Cuadrícula de evaluación (azul=acierto, rojo=fallo) ──────────────

def graficar_imagen(i, predicciones, etiquetas_reales, imagenes, clases):
    """Pinta una imagen con su predicción. Azul si acierta, rojo si falla."""
    pred, real, img = predicciones[i], etiquetas_reales[i], imagenes[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    clase_pred = np.argmax(pred)
    color = 'blue' if clase_pred == real else 'red'

    plt.xlabel("{} {:.0f}% ({})".format(
        clases[clase_pred],
        100 * np.max(pred),
        clases[real]
    ), color=color)


def graficar_evaluacion(modelo, pool, clases, filas=5, columnas=5):
    """Cuadrícula 5x5 con predicciones coloreadas."""
    n = filas * columnas
    indices = random.sample(range(len(pool)), n)

    # Preparar batch
    imagenes = np.empty((n, *CNN_TARGET_SIZE, 3), dtype=np.float32)
    etiquetas = np.zeros(n, dtype=int)
    for i, idx in enumerate(indices):
        jpeg_bytes, clase_id = pool[idx]
        imagenes[i] = decodificar(jpeg_bytes)
        etiquetas[i] = clase_id

    predicciones = modelo.predict(imagenes, verbose=0)

    # Pintar cuadrícula
    plt.figure(figsize=(2 * columnas, 2 * filas))
    for i in range(n):
        plt.subplot(filas, columnas, i + 1)
        graficar_imagen(i, predicciones, etiquetas, imagenes, clases)

    plt.suptitle("Evaluación CNN — Predicciones", fontsize=14)
    plt.tight_layout()
    plt.savefig('vis_evaluacion_cnn.png', dpi=150)
    plt.show()
    print("  Guardado: vis_evaluacion_cnn.png")


# ══════════════════════════════════════════════════════════════════════
# Flujo principal
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    RUTA_MODELO = 'modelo_final_habitaciones.keras'

    # 1. Cargar modelo
    print(f"Cargando modelo: {RUTA_MODELO}")
    modelo = load_model(RUTA_MODELO)

    # 2. Descargar imágenes
    print("Descargando imagenes...")
    pool = descargar_imagenes(CLASES_IMAGENES)

    # 3. Cuadrícula del pool
    print("Generando cuadrícula del pool...")
    graficar_pool(pool, CLASES_IMAGENES)

    # 4. Cuadrícula de evaluación
    print("Generando evaluación visual...")
    graficar_evaluacion(modelo, pool, CLASES_IMAGENES)

    print("Visualizaciones completadas.")
