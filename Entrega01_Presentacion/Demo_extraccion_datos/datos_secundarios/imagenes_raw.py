import kagglehub
import shutil
import os


def organizar_datasets():
    # 1. Diccionario con los datasets y el nombre de carpeta que queremos
    datasets = {
        "galinakg/interior-design-images-and-metadata": "data_estilos_5",
        "shiva12msk/interior-insight-modern-vs-old-classification1": "data_moderno_v_old",
        "shiva12msk/interior-insight-modern-vs-old-classification": "data_estancias_casa"
    }

    # Carpeta base donde guardaremos todo en el servidor
    base_path = os.path.join(os.getcwd(), "datasets_ia")

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for slug, nombre_local in datasets.items():
        print(f"\n--- Procesando: {slug} ---")

        # Descarga a la caché
        path_cache = kagglehub.dataset_download(slug)

        # Ruta de destino final
        destino_final = os.path.join(base_path, nombre_local)

        if not os.path.exists(destino_final):
            print(f"Moviendo a: {destino_final}...")
            # Movemos el contenido de la caché a nuestra carpeta del servidor
            shutil.move(path_cache, destino_final)
            print(f"✓ {nombre_local} listo.")
        else:
            print(f"¡Aviso! La carpeta {nombre_local} ya existe. Saltando...")

    print("\n" + "=" * 30)
    print("¡Todos los datasets están listos en el servidor!")
    print(f"Ruta: {base_path}")


if __name__ == "__main__":
    organizar_datasets()