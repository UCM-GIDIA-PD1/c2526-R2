from utils.funciones_minio import crear_cliente_minio, bajar_minio
from utils.config import PATH_DATASETS_MODELOS
import pandas as pd
from sklearn.model_selection import train_test_split

def bajar_df_texto() -> pd.DataFrame:
    """Baja el dataframe para los modelos de texto

    Returns:
        pd.DataFrame: Dataframe con las siguientes columnas
            id:             identificador único
            Descripcion:    (X) el texto
            Anuncia:        (y) variable categórica (Promotora|Intermediario|Particular)
            venta:          variable binaria que indica si el anuncio es de venta
            alquiler:       variable binaria que indica si el anuncio es de alquiler
    """
    cliente = crear_cliente_minio()
    objeto = "texto.parquet"
    df = bajar_minio(client=cliente, path=PATH_DATASETS_MODELOS, minio_object=objeto)
    return df

def separar_texto_train_test(df: pd.DataFrame):
    """
    Separa un DataFrame en variables de entrada (X) y objetivo (y),
    y divide los datos en conjuntos de entrenamiento y test.

    Asume que:
    - La columna 'Descripcion' contiene el texto (features)
    - La columna 'Anuncia' contiene la variable objetivo (labels)

    Args:
        df (pd.DataFrame): DataFrame de entrada con al menos las columnas
                           'Descripcion' y 'Anuncia'.

    Returns:
        tuple:
            - X_train (pd.Series): textos de entrenamiento
            - X_test (pd.Series): textos de test
            - y_train (pd.Series): etiquetas de entrenamiento
            - y_test (pd.Series): etiquetas de test
    """
    X = df["Descripcion"].astype(str)
    y = df["Anuncia"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test