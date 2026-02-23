import pandas as pd

fichero_coles = "centros_educativos.csv"

df_coles = pd.read_csv(fichero_coles, sep=";", encoding="latin1", quotechar='"', na_values=["", " "], keep_default_na=True)

cols_eliminar = [
    "DESCRIPCION-ENTIDAD",
    "DESCRIPCION",
    "CODIGO-POSTAL",
    "COD-BARRIO",
    "BARRIO",
    "COD-DISTRITO",
    "DISTRITO",
    "HORARIO",
    "EQUIPAMIENTO",
    "CONTENT-URL",
    "PLANTA",
    "PUERTA",
    "ESCALERAS",
    "ORIENTACION",
    "LOCALIDAD",
    "PROVINCIA",
    "LATITUD",
    "LONGITUD",
    "TELEFONO",
    "FAX",
    "EMAIL",
    "TIPO"
]

def limpiar_df_coles(df_coles):
    df = df_coles.dropna(subset=["DISTRITO"])
    df = df[df["DISTRITO"] != "DISTRITO"] # Elimina los centros de municipios que no sean Madrid 

    df = df.drop(columns=cols_eliminar) # Columnas sin información relevante o mayoría de valores vacíos.
    return df

df_coles = limpiar_df_coles(df_coles)

df_coles.to_csv('centros_educativos_limpio.csv', index=False, encoding='utf-8-sig')