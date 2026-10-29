# ==============================================================================
# sklearn_loader.py · PASO 0 del experimento: carga determinista de los 6 datasets
# ==============================================================================
"""
Carga los 6 conjuntos de datos del estudio (3 de clasificación y 3 de regresión), todos nativos
de scikit-learn, en dos variantes:
    "completo"       -> sin valores faltantes (fugas de ESCALADO y SELECCION)
    "con_faltantes"  -> 10 % de celdas vacías con semilla fija (fugas de IMPUTACION y COMBINADA)

Ejecución (prueba rápida de formas y faltantes):
    python 01_data/sklearn_loader.py

Lo importan: 02_pipelines/controlled_pairs/*.py, 04_experiments/*.py y 08_tests/test_project.py.
"""

# %% PASO 0 · Importaciones y parámetros globales
# ------------------------------------------------------------------------------
# QUÉ HACE   : importa los cargadores nativos de scikit-learn y fija los parámetros del diseño.
# POR QUÉ    : todos los datos salen de sklearn.datasets (observación R5 del comité): no hay
#              archivos externos ni anotadores humanos; cualquiera puede reproducir los datos.
# PARÁMETROS :
#   SEMILLA        = 42   -> semilla única del proyecto: dos ejecuciones dan datos idénticos.
#   N_RUIDO        = 50   -> columnas de ruido gaussiano añadidas a D1–D5 (ver PASO 3).
#   FRAC_FALTANTES = 0.10 -> fracción de celdas borradas en la variante "con_faltantes".
#   N_CALIFORNIA   = 3000 -> california_housing tiene 20.640 filas; KNNImputer cuesta ~n², por lo
#                            que se usa una submuestra determinista (desviación B1 del decision_log).
# ------------------------------------------------------------------------------
import numpy as np
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_digits,
    load_diabetes, fetch_california_housing, make_friedman1,
)

SEMILLA = 42
N_RUIDO = 50
FRAC_FALTANTES = 0.10
N_CALIFORNIA = 3000


# %% PASO 1 · Catálogo de datasets
# ------------------------------------------------------------------------------
# QUÉ HACE   : asocia cada identificador (D1…D6) con su nombre y su tarea.
# POR QUÉ    : la tarea decide la métrica (ROC-AUC / R²), el tipo de pliegues (estratificados o no)
#              y la versión del modelo (Classifier / Regressor) en todo el proyecto.
# ORIGEN     : D1 breast_cancer (569×30), D2 wine (178×13), D3 digits (1.797×64) -> clasificación;
#              D4 diabetes (442×10), D5 california_housing (submuestra 3.000×8),
#              D6 make_friedman1 (sintético 1.000×50, 45 columnas son ruido) -> regresión.
# ------------------------------------------------------------------------------
DATASETS = {
    "D1": ("breast_cancer",       "clasificacion"),
    "D2": ("wine",                "clasificacion"),
    "D3": ("digits",              "clasificacion"),
    "D4": ("diabetes",            "regresion"),
    "D5": ("california_housing",  "regresion"),
    "D6": ("friedman1",           "regresion"),
}


# %% PASO 2 · Carga de los datos originales
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: funciones load_* de scikit-learn (vienen dentro de la librería, sin internet),
#   salvo fetch_california_housing, que se descarga una sola vez y queda en la caché local de
#   scikit-learn; make_friedman1 se genera con semilla fija.
# LÓGICA     : devuelve (X, y) tal como los publica scikit-learn. Para D5 elige siempre las mismas
#              3.000 filas (muestreo sin reemplazo con SEMILLA), de modo que el submuestreo es
#              reproducible y no introduce variabilidad entre corridas.
# USO        : cargar_base() es la versión pública que usa el EDA (datos sin ruido ni faltantes).
# ------------------------------------------------------------------------------
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
        filas = np.random.default_rng(SEMILLA).choice(len(y), N_CALIFORNIA, replace=False)
        return X[filas], y[filas]
    if id_dataset == "D6":
        return make_friedman1(n_samples=1000, n_features=50, noise=1.0, random_state=SEMILLA)
    raise ValueError(f"Dataset desconocido: {id_dataset}. Opciones: {list(DATASETS)}")


def cargar_base(id_dataset):
    """Versión pública de _cargar_base: datos originales, sin ruido ni faltantes (para el EDA)."""
    return _cargar_base(id_dataset)


# %% PASO 3 · Preparación trazable: ruido controlado y valores faltantes
# ------------------------------------------------------------------------------
# QUÉ HACE   : convierte los datos originales en la variante pedida por el experimento.
# LÓGICA (en este orden, que NO debe alterarse porque cambiaría los números aleatorios):
#   1) Generador propio por dataset: semilla = 42 + posición del dataset en el catálogo. Así,
#      agregar o quitar un dataset no cambia el ruido ni los faltantes de los demás.
#   2) Ruido: se añaden n_ruido columnas N(0, 1) a D1–D5. ¿Por qué? La fuga por selección solo se
#      manifiesta si hay variables inútiles entre las cuales "elegir": si SelectKBest elige con las
#      etiquetas de TODO el dataset, encuentra por azar columnas de ruido que parecen informativas.
#      D6 ya trae 45 columnas de ruido, por eso no se le agregan más.
#   3) Faltantes (solo "con_faltantes"): cada celda se borra con probabilidad 0,10 (MCAR: faltante
#      completamente al azar). Es lo que el imputador tendrá que rellenar.
# SALIDA     : X (float, n_filas × n_columnas) e y (etiquetas o valores continuos).
# INTERPRETACIÓN: el ruido y los faltantes son simulados; por eso los resultados describen el
#   mecanismo de la fuga en condiciones controladas, no la calidad de datos de un caso real.
# ------------------------------------------------------------------------------
def cargar_dataset(id_dataset, variante="completo", n_ruido=None):
    """
    Carga un dataset del estudio.

    Parámetros
    ----------
    id_dataset : "D1" ... "D6"
    variante   : "completo" | "con_faltantes"
    n_ruido    : columnas de ruido a añadir (por defecto N_RUIDO; noise_sensitivity.py lo varía)

    Devuelve
    --------
    X : ndarray (n_filas, n_columnas) de float; y : ndarray de etiquetas o valores.
    """
    if variante not in ("completo", "con_faltantes"):
        raise ValueError("variante debe ser 'completo' o 'con_faltantes'")

    X, y = _cargar_base(id_dataset)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # 1) Generador independiente por dataset.
    indice = list(DATASETS).index(id_dataset)
    rng = np.random.default_rng(SEMILLA + indice)

    # 2) Columnas de ruido gaussiano (excepto D6, que ya las trae).
    n_ruido = N_RUIDO if n_ruido is None else n_ruido
    if id_dataset != "D6" and n_ruido > 0:
        ruido = rng.normal(size=(X.shape[0], n_ruido))
        X = np.hstack([X, ruido])

    # 3) Valores faltantes (solo en la variante "con_faltantes").
    if variante == "con_faltantes":
        mascara = rng.random(X.shape) < FRAC_FALTANTES
        X = X.copy()
        X[mascara] = np.nan

    return X, y


def tarea_de(id_dataset):
    """'clasificacion' o 'regresion' según el dataset."""
    return DATASETS[id_dataset][1]


# %% PASO 4 · Verificación rápida
# ------------------------------------------------------------------------------
# QUÉ HACE   : imprime, por dataset, la forma de X y la fracción real de faltantes de la variante.
# CÓMO LEERLO: X debe tener las columnas originales + 50 de ruido (D6: 50 en total) y la fracción
#              de faltantes debe rondar 0,100 (variación muestral esperable: ±0,005).
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    for ident, (nombre, tarea) in DATASETS.items():
        X, y = cargar_dataset(ident)
        Xf, _ = cargar_dataset(ident, "con_faltantes")
        print(f"{ident} {nombre:20s} {tarea:14s} X={X.shape}  faltantes(variante)={np.isnan(Xf).mean():.3f}")
