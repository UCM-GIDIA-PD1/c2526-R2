import os
import sys
import subprocess

def buscar_modulos_pipeline(directorio_base):
    """
    Busca todas las carpetas dentro del directorio base que contengan un archivo 'main.py'.
    """
    carpetas_pipeline = []
    
    # Recorrer los elementos en el directorio base
    for elemento in os.listdir(directorio_base):
        ruta_elemento = os.path.join(directorio_base, elemento)
        
        if os.path.isdir(ruta_elemento):
            # Verificar si dentro de la carpeta existe un 'main.py'
            ruta_main = os.path.join(ruta_elemento, "main.py")
            if os.path.exists(ruta_main):
                carpetas_pipeline.append(elemento)
                
    carpetas_pipeline.sort()
    
    return carpetas_pipeline

def menu_principal(lista_carpetas):
    """
    Muestra el menú principal para seleccionar en qué módulo del pipeline entrar.
    """
    while True:
        print("\n" + "=" * 50)
        print(" MENU PRINCIPAL - PIPELINE DE DATOS")
        print("=" * 50)

        for i, carpeta in enumerate(lista_carpetas):
            print(f"  {i + 1}. Fase: {carpeta}")

        print("-" * 50)
        print(f"  0. Salir de la aplicación")
        print("=" * 50)

        opcion = input("  Elige el módulo al que deseas acceder: ")

        try:
            opcion_int = int(opcion)

            if opcion_int == 0:
                print("\n  Cerrando orquestador principal. ¡Hasta luego!")
                return None

            if 1 <= opcion_int <= len(lista_carpetas):
                carpeta_elegida = lista_carpetas[opcion_int - 1]
                return carpeta_elegida
            else:
                print(f"\n  Error: Por favor, elige un número entre 0 y {len(lista_carpetas)}.")

        except ValueError:
            print("\n  Error: Entrada no válida. Tienes que escribir un número.")

def ejecutar_modulo(carpeta_modulo: str, directorio_base: str):
    """
    Ejecuta el archivo main.py dentro de la carpeta seleccionada.
    Cambia el directorio de trabajo (cwd) temporalmente para que el sub-script funcione correctamente.
    """
    ruta_carpeta_modulo = os.path.join(directorio_base, carpeta_modulo)
    subprocess.run([sys.executable, "main.py"], cwd=ruta_carpeta_modulo)

if __name__ == '__main__':
    # Obtener el directorio raíz (donde se encuentra este orquestador)
    directorio_base = os.path.dirname(os.path.abspath(__file__))
    
    # Encontrar módulos disponibles
    modulos = buscar_modulos_pipeline(directorio_base)
    
    if not modulos:
        print(f"Error: No se encontraron carpetas con un archivo 'main.py' en '{directorio_base}'.")
    else:
        # Bucle principal
        modulo_elegido = menu_principal(modulos)
        while modulo_elegido is not None:
            ejecutar_modulo(modulo_elegido, directorio_base)
            # Al salir del sub-menú, vuelve a mostrar el menú principal
            modulo_elegido = menu_principal(modulos)