import pandas as pd


#Estos datos no requieren de API o Web Scrapping, ya que son de acceso público en la Comunidad de Madrid en la siguiente página: https://datos.madrid.es/egob/catalogo/212769-0-atencion-medica.csv
DATASET_PATH = "212769-0-atencion-medica.csv"

COLUMNAS_A_ELIMINAR = [
    "DESCRIPCION",
    "HORARIO",
    "EQUIPAMIENTO",
    "TRANSPORTE",
    "DESCRIPCION",
    "ACCESIBILIDAD",
    "CONTENT-URL",
    "NOMBRE-VIA",
    "CLASE-VIAL",
    "TIPO-NUM",
    "NUM",
    "PLANTA",
    "PUERTA",
    "ESCALERAS",
    "ORIENTACION",
    "TELEFONO",
    "FAX",
    "EMAIL"
]

OUTPUT_HEAD_PATH = "dataset_head_5.csv"


def main():

    
    df = pd.read_csv(
        DATASET_PATH,
        encoding="latin1",
        sep=";"
    )       
    print(df.columns.to_list())
    df = df.drop(columns=COLUMNAS_A_ELIMINAR, errors="ignore")

    df.to_csv(DATASET_PATH, index=False)

    df.head(5).to_csv(OUTPUT_HEAD_PATH, index=False)


if __name__ == "__main__":
    main()





