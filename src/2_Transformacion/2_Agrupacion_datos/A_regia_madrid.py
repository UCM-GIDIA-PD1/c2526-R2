import geopandas as gpd
import matplotlib.pyplot as plt
import time
import folium
import matplotlib
import mapclassify

def visualizar_shapefile_barrios(ruta_shp):
    """
    Carga un archivo Shapefile (.shp), imprime su información básica 
    y lo dibuja en pantalla para comprobar su calidad.
    """
    print(f"📂 Cargando el shapefile desde: {ruta_shp}")
    
    # 1. CARGAR EL ARCHIVO
    # GeoPandas lee el .shp y todos sus archivos hermanos (.dbf, .shx, .prj) automáticamente
    gdf_barrios = gpd.read_file(ruta_shp)
    
    # 2. EXPLORACIÓN DE DATOS (Vital para saber qué tenemos)
    print("\n🔍 Exploración rápida de los datos:")
    print(f"   - Total de polígonos (barrios/distritos): {len(gdf_barrios)}")
    print(f"   - Sistema de Coordenadas (CRS): {gdf_barrios.crs}")
    print(f"   - Columnas disponibles: {list(gdf_barrios.columns)}")
    
    # Mostramos las 5 primeras filas (sin la columna de geometría que ocupa mucho)
    # Esto te ayudará a identificar qué columna tiene el nombre del barrio
    columnas_sin_geom = [col for col in gdf_barrios.columns if col != 'geometry']
    print("\n📝 Muestra de los primeros 3 registros:")
    print(gdf_barrios[columnas_sin_geom].head(30))
    print(gdf_barrios["geometry"][:30])
    # 3. DIBUJAR EL MAPA
    print("\n🎨 Generando la visualización...")
    
    # Creamos un lienzo grande (12x12 pulgadas) para verlo bien
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Dibujamos los polígonos
    # - edgecolor: color del borde (las fronteras del barrio)
    # - facecolor: color de relleno (usamos 'none' para que sea transparente o un color claro)
    # - linewidth: grosor de la línea
    gdf_barrios.plot(
        ax=ax, 
        edgecolor='black', 
        facecolor='lightblue', 
        alpha=0.6, 
        linewidth=0.8
    )
    
    # Le ponemos título y quitamos los ejes (los números de los bordes) para que parezca un mapa real
    plt.title("Visualización del Shapefile: Barrios de Madrid", fontsize=16, fontweight='bold')
    ax.set_axis_off()
    
    # Mostramos la ventana
    plt.tight_layout()
    plt.show()

def comprobar_tipo_geometria(ruta_shp):
    """
    Lee un archivo espacial y te dice exactamente qué tipo de formas geométricas 
    (Puntos, Líneas o Polígonos) contiene y cuántas hay de cada una.
    """
    print(f"📂 Cargando archivo: {ruta_shp}")
    gdf = gpd.read_file(ruta_shp)
    
    print("\n🔍 Analizando las formas geométricas...")
    
    # .geom_type nos dice el tipo de forma de cada fila. 
    # Usamos value_counts() para agruparlas y contarlas.
    tipos_de_formas = gdf.geom_type.value_counts()
    
    print("=====================================")
    print("📊 RESULTADO DEL ANÁLISIS DE FORMAS:")
    print("=====================================")
    for forma, cantidad in tipos_de_formas.items():
        print(f" - {forma}: {cantidad} elementos")
    print("=====================================")
    
    # 💡 Lógica de diagnóstico automático:
    if "LineString" in tipos_de_formas or "MultiLineString" in tipos_de_formas:
        print("\n⚠️ DIAGNÓSTICO: Este archivo dibuja LÍNEAS. Es probablemente un mapa de calles (callejero), carreteras o ríos.")
        print("❌ NO sirve para agrupar pisos por barrios.")
        
    elif "Polygon" in tipos_de_formas or "MultiPolygon" in tipos_de_formas:
        print("\n✅ DIAGNÓSTICO: Este archivo dibuja ÁREAS CERRADAS (Polígonos).")
        print("🎯 ¡PERFECTO! Sirve para agrupar pisos por barrios o distritos.")
        
    elif "Point" in tipos_de_formas or "MultiPoint" in tipos_de_formas:
        print("\n⚠️ DIAGNÓSTICO: Este archivo dibuja PUNTOS exactos.")
        print("📍 Son ubicaciones individuales (como tus pisos o los supermercados).")

# ==========================================
# 🧪 CÓMO USARLO:
# ==========================================
# Cambia esta ruta por la ruta real donde hayas guardado tu archivo .shp
# ¡Ojo! Asegúrate de que en esa misma carpeta estén los archivos .shx y .dbf que vienen con él.
ruta_mi_archivo = "C:/Users/harra/gdie_callejero_compl_V223.shp"
comprobar_tipo_geometria(ruta_mi_archivo)
visualizar_shapefile_barrios(ruta_mi_archivo)