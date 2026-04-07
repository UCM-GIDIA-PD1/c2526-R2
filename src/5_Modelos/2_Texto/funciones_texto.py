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

def x_y_split(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Separa el DataFrame de original en variable de entrada (x) y salida (y)
    Asume que:
    - La columna 'Descripcion' contiene el texto (features)
    - La columna 'Anuncia' contiene la variable objetivo (labels)

    Args:
        df (pd.DataFrame): DataFrame de entrada con al menos las columnas
                           'Descripcion' y 'Anuncia'.

    Returns:
        tuple[x: pd.Series, y: pd.Series]: 
        - x (pd.Series): variable de entrada (textos de los anuncios)
        - y (pd.Series): variable objetivo
    """
    x = df["Descripcion"].astype(str)
    y = df["Anuncia"].astype(str)
    return x, y

def train_val_test_split(x: pd.Series, y: pd.Series) -> tuple:
    """_summary_

    Args:
        x (pd.Series): Textos de anuncios
        y (pd.Series): Etiquetas (Particular, Intermediario o Promotora)

    Returns:
        tuple: x_train, x_val, x_test, y_train, y_val, y_test
    """
    

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )


    return x_train, x_val, x_test, y_train, y_val, y_test