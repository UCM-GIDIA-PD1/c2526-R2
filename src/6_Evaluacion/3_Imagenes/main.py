import os

def main():
    notebook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Imagenes.ipynb")
    print("\n" + "=" * 50)
    print("  EVALUACIÓN DE MODELOS - IMÁGENES")
    print("=" * 50)
    print("\n  La evaluación de modelos de imágenes se realiza")
    print("  en un Jupyter Notebook interactivo.")
    print(f"\n  Notebook: {notebook_path}")
    print("\n  Ábrelo con:")
    print("    jupyter notebook Imagenes.ipynb")
    print("  o desde VS Code abriendo el archivo .ipynb directamente.")
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
