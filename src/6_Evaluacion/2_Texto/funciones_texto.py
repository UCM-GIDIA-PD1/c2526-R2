from utils.funciones_minio import crear_cliente_minio, bajar_minio
from utils.config import PATH_DATASETS_MODELOS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
import wandb
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, confusion_matrix,
)


MODEL_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"


class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):
    """Wrapper sklearn-compatible para sentence-transformers.

    Convierte textos en vectores de embeddings densos (384 dimensiones) usando
    un modelo pre-entrenado. Compatible con sklearn Pipelines y GridSearchCV.

    Args:
        model_name (str): Nombre del modelo en HuggingFace / sentence-transformers.
        batch_size (int): Tamaño de batch para la codificación.
    """

    def __init__(
        self,
        model_name: str = MODEL_EMBEDDINGS,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._encoder = None

    def _get_encoder(self) -> SentenceTransformer:
        """Carga el modelo de forma perezosa (solo la primera vez)."""
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def fit(self, X, y=None):
        self._get_encoder()
        return self

    def transform(self, X) -> np.ndarray:
        encoder = self._get_encoder()
        return encoder.encode(
            list(X),
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )


def obtener_o_cargar_embeddings(
    X_train: pd.Series,
    X_val: pd.Series,
    X_test: pd.Series,
    model_name: str = MODEL_EMBEDDINGS,
    batch_size: int = 32,
    cache_dir: str = "embeddings_cache",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve los embeddings de los tres splits, usando caché en disco si existe.

    La primera vez que se llama, codifica los textos con sentence-transformers y
    guarda los arrays en ``cache_dir`` como ficheros ``.npy``. Las siguientes
    ejecuciones (cualquier script del proyecto) los cargan directamente sin
    necesidad de volver a descargar el modelo ni re-codificar.

    El nombre de los ficheros incluye el modelo para evitar colisiones si se
    cambia de modelo sin borrar la caché manualmente.

    Args:
        X_train:   Textos de entrenamiento.
        X_val:     Textos de validación.
        X_test:    Textos de test.
        model_name: Nombre del modelo de sentence-transformers.
        batch_size: Tamaño de batch para la codificación.
        cache_dir:  Directorio donde se guardan/leen los ficheros ``.npy``.

    Returns:
        tuple: (emb_train, emb_val, emb_test), arrays numpy de shape (n, 384).
    """
    import os

    # Sufijo seguro para usar como parte del nombre de fichero
    model_slug = model_name.replace("/", "_").replace("-", "_")
    cache_path = os.path.join(cache_dir, model_slug)
    os.makedirs(cache_path, exist_ok=True)

    paths = {
        "train": os.path.join(cache_path, "emb_train.npy"),
        "val":   os.path.join(cache_path, "emb_val.npy"),
        "test":  os.path.join(cache_path, "emb_test.npy"),
    }

    cache_completa = all(os.path.exists(p) for p in paths.values())

    if cache_completa:
        print(f"✅ Cargando embeddings desde caché: {cache_path}")
        emb_train = np.load(paths["train"])
        emb_val   = np.load(paths["val"])
        emb_test  = np.load(paths["test"])
        print(f"   shape train: {emb_train.shape}")
    else:
        print(f"Caché no encontrada. Codificando con '{model_name}'...")
        encoder = SentenceTransformer(model_name)

        print("  Codificando train...")
        emb_train = encoder.encode(
            list(X_train), batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        print("  Codificando val...")
        emb_val = encoder.encode(
            list(X_val), batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        print("  Codificando test...")
        emb_test = encoder.encode(
            list(X_test), batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )

        np.save(paths["train"], emb_train)
        np.save(paths["val"],   emb_val)
        np.save(paths["test"],  emb_test)
        print(f"💾 Embeddings guardados en: {cache_path}")

    return emb_train, emb_val, emb_test


def bajar_df_texto() -> pd.DataFrame:
    """Baja el dataframe para los modelos de texto.

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
    """Separa el DataFrame en variable de entrada (x) y salida (y).

    Asume que:
    - La columna 'Descripcion' contiene el texto (features)
    - La columna 'Anuncia' contiene la variable objetivo (labels)

    Args:
        df (pd.DataFrame): DataFrame con al menos las columnas 'Descripcion' y 'Anuncia'.

    Returns:
        tuple[x: pd.Series, y: pd.Series]:
            - x: textos de los anuncios
            - y: variable objetivo
    """
    x = df["Descripcion"].astype(str)
    y = df["Anuncia"].astype(str)
    return x, y


def train_val_test_split(x: pd.Series, y: pd.Series) -> tuple:
    """Divide los datos en entrenamiento (80%), validación (10%) y test (10%).

    Args:
        x (pd.Series): Textos de anuncios.
        y (pd.Series): Etiquetas (Particular, Intermediario o Promotora).

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


def evaluar_modelo(model, X, y) -> dict:
    """Calcula las métricas estándar del proyecto sobre (X, y)."""
    y_pred = model.predict(X)
    return {
        "f1_macro": f1_score(y, y_pred, average="macro"),
        "accuracy": accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
    }


def analizar_por_longitud(model, X_test, y_test) -> pd.DataFrame:
    """Calcula F1-macro por segmento de longitud del texto (en palabras)."""
    df = pd.DataFrame({"texto": X_test.values, "y_true": y_test.values})
    df["longitud"] = df["texto"].apply(lambda x: len(x.split()))

    bins = [0, 20, 50, 100, np.inf]
    labels = ["corto", "medio", "largo", "muy_largo"]
    df["segmento"] = pd.cut(df["longitud"], bins=bins, labels=labels)

    resultados = []
    for seg in labels:
        subset = df[df["segmento"] == seg]
        if len(subset) == 0:
            continue
        y_pred = model.predict(subset["texto"])
        resultados.append({
            "segmento": seg,
            "n_samples": len(subset),
            "f1_macro": f1_score(subset["y_true"], y_pred, average="macro"),
        })
    return pd.DataFrame(resultados)


def loguear_resultados_test(model, X_test, y_test) -> dict:
    """Loguea en wandb TODAS las métricas de test con el esquema unificado.

    Esta función produce los mismos nombres de métricas y tablas para
    cualquier modelo del proyecto, lo que permite comparar runs entre sí.
    """
    metricas = evaluar_modelo(model, X_test, y_test)
    y_pred = model.predict(X_test)
    labels = sorted(np.unique(y_test))

    # 1. Métricas globales en test
    wandb.log({
        "test/f1_macro": metricas["f1_macro"],
        "test/accuracy": metricas["accuracy"],
        "test/precision_macro": metricas["precision_macro"],
        "test/recall_macro": metricas["recall_macro"],
    })

    # 2. Métricas por clase
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    per_class_log = {}
    for clase in labels:
        if clase in report:
            per_class_log[f"test/per_class/{clase}/f1"] = report[clase]["f1-score"]
            per_class_log[f"test/per_class/{clase}/precision"] = report[clase]["precision"]
            per_class_log[f"test/per_class/{clase}/recall"] = report[clase]["recall"]
            per_class_log[f"test/per_class/{clase}/support"] = report[clase]["support"]
    wandb.log(per_class_log)

    # 3. Análisis por longitud
    df_long = analizar_por_longitud(model, X_test, y_test)
    by_length_log = {}
    for _, row in df_long.iterrows():
        by_length_log[f"test/by_length/{row['segmento']}/f1_macro"] = row["f1_macro"]
        by_length_log[f"test/by_length/{row['segmento']}/n_samples"] = row["n_samples"]
    wandb.log(by_length_log)

    # 4. Matriz de confusión (wandb espera ÍNDICES enteros, no strings)
    class_to_idx = {c: i for i, c in enumerate(labels)}
    y_true_idx = [class_to_idx[c] for c in np.asarray(y_test)]
    y_pred_idx = [class_to_idx[c] for c in np.asarray(y_pred)]

    wandb.log({
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true_idx,
            preds=y_pred_idx,
            class_names=[str(l) for l in labels],
        )
    })
    

    # 5. Tablas auxiliares
    wandb.log({"test/by_length_table": wandb.Table(dataframe=df_long)})
    report_df = (
        pd.DataFrame(report).T.reset_index().rename(columns={"index": "clase"})
    )
    wandb.log({"test/classification_report": wandb.Table(dataframe=report_df)})

    # Para imprimir en consola
    print("\n=== RESULTADOS EN TEST ===")
    print(metricas)
    print("\n=== ANALISIS POR CLASE ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\n=== ANALISIS POR LONGITUD ===")
    print(df_long)
    print("\n=== MATRIZ DE CONFUSION ===")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=labels, columns=labels))

    return metricas