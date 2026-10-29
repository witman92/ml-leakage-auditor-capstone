"""
sklearn_loader.py  ·  PASO 0 del experimento: carga determinista de los 6 datasets.

¿Qué hace este archivo?
    Entrega los 6 conjuntos de datos del estudio (3 de clasificación y 3 de regresión),
    todos nativos de scikit-learn, listos para usar. No se necesita internet salvo para
    california_housing, que scikit-learn descarga una sola vez y deja en caché.

Dos variantes de cada dataset (para poder provocar cada tipo de fuga):
    "completo"        -> sin valores faltantes (sirve para fugas de ESCALADO y SELECCIÓN)
    "con_faltantes"   -> 10 % de celdas vacías puestas al azar con semilla fija
                         (sirve para fugas de IMPUTACIÓN y para la fuga COMBINADA)

Columnas de ruido:
    A los datasets D1–D5 se les agregan N_RUIDO columnas de ruido gaussiano puro.
    ¿Por qué? Porque la fuga por selección de variables solo se nota cuando hay variables
    inútiles entre las cuales "elegir": si se elige con las etiquetas de TODO el dataset,
    el selector encuentra por azar columnas de ruido que parecen informativas.
    D6 (make_friedman1) ya trae 45 columnas de ruido, por eso no se le agregan más.

Todo usa la semilla SEMILLA = 42: dos ejecuciones dan exactamente los mismos datos.
"""

import numpy as np
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_digits,
    load_diabetes, fetch_california_housing, make_friedman1,
)

SEMILLA = 42          # semilla única de todo el proyecto
N_RUIDO = 50          # columnas de ruido añadidas a D1–D5
FRAC_FALTANTES = 0.10  # fracción de celdas que se borran en la variante "con_faltantes"
N_CALIFORNIA = 3000   # california_housing tiene 20.640 filas; se submuestrean 3.000 para
                      # que KNNImputer (costo ~ n²) sea ejecutable en un portátil.

# Catálogo: id -> (nombre, tarea). La tarea decide métricas, tipo de validación y modelos.
DATASETS = {
    "D1": ("breast_cancer",       "clasificacion"),
    "D2": ("wine",                "clasificacion"),
    "D3": ("digits",              "clasificacion"),
    "D4": ("diabetes",            "regresion"),
    "D5": ("california_housing",  "regresion"),
    "D6": ("friedman1",           "regresion"),
}


def _cargar_base(id_dataset):
    """Devuelve (X, y) originales de scikit-learn, sin ruido ni faltantes."""
    if id_dataset == "D1":
        return load_breast_cancer(return_X_y=True)
    if id_dataset == "D2":
        return load_wine(return_X_y=True)
    if id_dataset == "D3":
        return load_digits(return_X_y=True)
    if id_dataset == "D4":
        return load_diabetes(return_X_y=True)
    if id_dataset == "D5":
        X, y = fetch_california_housing(return_X_y=True)
        # Submuestreo determinista: siempre las mismas 3.000 filas.
        filas = np.random.default_rng(SEMILLA).choice(len(y), N_CALIFORNIA, replace=False)
        return X[filas], y[filas]
    if id_dataset == "D6":
        return make_friedman1(n_samples=1000, n_features=50, noise=1.0, random_state=SEMILLA)
    raise ValueError(f"Dataset desconocido: {id_dataset}. Opciones: {list(DATASETS)}")


def cargar_base(id_dataset):
    """Versión pública de _cargar_base: datos originales, sin ruido ni faltantes (para el EDA)."""
    return _cargar_base(id_dataset)


def cargar_dataset(id_dataset, variante="completo", n_ruido=None):
    """
    Carga un dataset del estudio.

    Parámetros
    ----------
    id_dataset : "D1" ... "D6"
    variante   : "completo" | "con_faltantes"
    n_ruido    : columnas de ruido a añadir (por defecto N_RUIDO; el análisis de sensibilidad lo varía)

    Devuelve
    --------
    X : ndarray (n_filas, n_columnas) de float; y : ndarray de etiquetas o valores.
    """
    if variante not in ("completo", "con_faltantes"):
        raise ValueError("variante debe ser 'completo' o 'con_faltantes'")

    X, y = _cargar_base(id_dataset)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # Cada dataset usa su propio generador (semilla + número del dataset) para que
    # añadir o quitar un dataset no cambie el ruido de los demás.
    indice = list(DATASETS).index(id_dataset)
    rng = np.random.default_rng(SEMILLA + indice)

    # 1) Columnas de ruido gaussiano (excepto D6, que ya las trae).
    n_ruido = N_RUIDO if n_ruido is None else n_ruido
    if id_dataset != "D6" and n_ruido > 0:
        ruido = rng.normal(size=(X.shape[0], n_ruido))
        X = np.hstack([X, ruido])

    # 2) Valores faltantes (solo en la variante "con_faltantes").
    if variante == "con_faltantes":
        mascara = rng.random(X.shape) < FRAC_FALTANTES
        X = X.copy()
        X[mascara] = np.nan

    return X, y


def tarea_de(id_dataset):
    """'clasificacion' o 'regresion' según el dataset."""
    return DATASETS[id_dataset][1]


if __name__ == "__main__":
    # Prueba rápida:  python 01_data/sklearn_loader.py
    for ident, (nombre, tarea) in DATASETS.items():
        X, y = cargar_dataset(ident)
        Xf, _ = cargar_dataset(ident, "con_faltantes")
        print(f"{ident} {nombre:20s} {tarea:14s} X={X.shape}  faltantes(variante)={np.isnan(Xf).mean():.3f}")
