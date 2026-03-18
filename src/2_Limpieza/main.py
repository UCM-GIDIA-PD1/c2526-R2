import os
import sys
import subprocess

def buscar_archivos_en_carpeta(carpeta):
    """
    Ejecuta todos los archivos Python (.py) que se encuentran en la carpeta especificada,
    excluyendo el archivo 'main.py'.
    
    Parameters:
    carpeta (src/1_Extracción): La ruta del directorio en el que se buscarán los archivos Python.
    """
    # Listar todos los archivos en la carpeta
    archivos = [archivo for archivo in os.listdir(carpeta) if archivo.endswith('.py') and archivo != 'main.py']

    archivos.sort()

    return archivos

def menu_seleccion(lista_archivos):
    """
    Muestra un menú interactivo, controla errores de entrada 
    y devuelve el nombre del archivo seleccionado.
    """
    while True:
        print("\n" + "="*40)
        print("  MENÚ PIPELINE LIMPIEZA")
        print("="*40)
        
        for i, archivo in enumerate(lista_archivos):
            print(f"  {i + 1}. {archivo}")
            
        print("-" * 40)
        print(f"  0. Salir de la limpieza")
        print("="*40)
        
        opcion = input("  Elige el número del script a ejecutar: ")
        
        try:
            opcion_int = int(opcion)
            
            if opcion_int == 0:
                print("\n  Saliendo del menú...")
                return None
                
            if 1 <= opcion_int <= len(lista_archivos):
                archivo_elegido = lista_archivos[opcion_int - 1]
                print(f"\n  Has seleccionado: {archivo_elegido}")
                return archivo_elegido
            else:
                print(f"\n  Error: Por favor, elige un número entre 0 y {len(lista_archivos)}.")
                
        except ValueError:
            print("\n  Error: Entrada no válida. Tienes que escribir un número.")

def ejecutar_archivo(archivo:str,carpeta:str):
    ruta_completa = os.path.join(carpeta, archivo)
    subprocess.run([sys.executable, ruta_completa])

if __name__ == '__main__':
    """
    Punto de entrada principal del script. Obtiene la ruta de la carpeta en la que
    se encuentra 'main.py' y ejecuta todos los archivos Python (.py) en esa carpeta,
    excluyendo 'main.py'.
    """
    # Obtener la carpeta donde se encuentra el archivo main.py
    carpeta = os.path.dirname(os.path.abspath(__file__))  
    
    # Verificar si la carpeta existe
    if not os.path.exists(carpeta):
        print(f"Error: El directorio '{carpeta}' no existe.")
    else:
        ejecutables = buscar_archivos_en_carpeta(carpeta)
        ejecutable = menu_seleccion(ejecutables)
        while ejecutable != None:
            ejecutar_archivo(ejecutable,carpeta)
            ejecutable = menu_seleccion(ejecutables)
