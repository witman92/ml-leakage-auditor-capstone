# ==============================================================================
# generate_controlled_pairs.py · PASO 1 del experimento: banco de control (oráculo programático)
# ==============================================================================
"""
Escribe, para cada combinación  dataset (6) × familia de modelo (4) × tipo de fuga (4), DOS scripts
de Python: uno con fuga ("leaky") y otro idéntico sin fuga ("clean").
Total: 6 × 4 × 4 = 96 pares = 192 archivos en 02_pipelines/controlled_pairs/.

Ejecución:
    python 02_pipelines/generate_controlled_pairs.py

Lo importan: 04_experiments/run_benchmark.py, 02_pipelines/llm_generated/generate_with_llm.py
y 08_tests/test_project.py.
"""

# %% PASO 0 · Por qué este banco reemplaza a los anotadores humanos
# ------------------------------------------------------------------------------
# ORÁCULO PROGRAMÁTICO: sabemos con certeza qué script tiene fuga porque lo construimos así. Esa
#   etiqueta ("verdad conocida", ground truth) permite medir precisión y recall del auditor sin
#   depender de opiniones (observación R5 del comité).
# DISEÑO PAREADO: lo ÚNICO que difiere entre "leaky" y "clean" es dónde se ajustan los
#   transformadores:
#     leaky : se ajustan con TODAS las filas, ANTES de la validación cruzada.
#     clean : van dentro de un sklearn.pipeline.Pipeline y se ajustan por pliegue.
#   Mismo dataset, mismos pliegues (semilla 42), mismo modelo, mismos hiperparámetros. Por eso la
#   diferencia de puntaje entre ambos (ΔOpt) se atribuye únicamente a la fuga.
# LIMITACIÓN: el generador y el auditor tienen el mismo autor (evaluación circular); por eso existe
#   la evaluación independiente con 13 scripts escritos a mano (08_tests/handwritten_cases).
# ------------------------------------------------------------------------------

# %% PASO 1 · Catálogos del diseño factorial
# ------------------------------------------------------------------------------
# QUÉ HACE   : define los tres factores del diseño (datasets, familias, tipos de fuga).
# ORIGEN     : los datasets se cargan en tiempo de ejecución desde 01_data/sklearn_loader.py
#              (aquí solo se guarda su id, tarea y nombre para escribir el código de cada script).
# FAMILIAS   : RF (ensamble de árboles), GB (boosting moderno), LIN (lineal regularizado L2) y
#              MLP (red neuronal). Cada una tiene versión de clasificación y de regresión, con
#              hiperparámetros fijos y random_state=42.
# FUGAS      : (variante de datos, pasos). Cada paso = (nombre, constructor, ¿usa y?).
#              ESCALADO   -> StandardScaler ajustado con todo el dataset.
#              IMPUTACION -> KNNImputer ajustado con todo el dataset (vecinos de prueba).
#              SELECCION  -> SelectKBest elige variables mirando las etiquetas de TODAS las filas.
#              COMBINADA  -> imputación + escalado + selección a la vez.
#              {F} se reemplaza por f_classif (clasificación) o f_regression (regresión).
# ESCALADO INTERNO: LIN y MLP necesitan variables en escalas comparables. En IMPUTACION y
#              SELECCION (donde la fuga NO es el escalado) llevan un StandardScaler DENTRO del
#              modelo, idéntico en leaky y clean, para que el escalado no sea fuente de
#              inestabilidad ni de diferencia entre versiones (corrección C1 del decision_log).
# ------------------------------------------------------------------------------
from pathlib import Path

CARPETA_SALIDA = Path(__file__).resolve().parent / "controlled_pairs"
DATASETS = {"D1": "clasificacion", "D2": "clasificacion", "D3": "clasificacion",
            "D4": "regresion", "D5": "regresion", "D6": "regresion"}
NOMBRE_DATASET = {"D1": "breast_cancer", "D2": "wine", "D3": "digits",
                  "D4": "diabetes", "D5": "california_housing", "D6": "friedman1"}

# Familia -> (importación, constructor en clasificación, constructor en regresión)
FAMILIAS = {
    "RF": ("from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor",
           "RandomForestClassifier(n_estimators=100, random_state=42)",
           "RandomForestRegressor(n_estimators=100, random_state=42)"),
    "GB": ("from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor",
           "HistGradientBoostingClassifier(random_state=42)",
           "HistGradientBoostingRegressor(random_state=42)"),
    # LogisticRegression ya es L2 por defecto; max_iter alto evita avisos de no convergencia.
    "LIN": ("from sklearn.linear_model import LogisticRegression, Ridge",
            "LogisticRegression(C=1.0, max_iter=2000, random_state=42)",
            "Ridge(alpha=1.0, random_state=42)"),
    "MLP": ("from sklearn.neural_network import MLPClassifier, MLPRegressor",
            "MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)",
            "MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)"),
}

# Tipo de fuga -> (variante de datos, lista de pasos). Cada paso = (nombre, constructor, usa_y).
# {F} se reemplaza por f_classif (clasificación) o f_regression (regresión).
FUGAS = {
    "ESCALADO":   ("completo",      [("escalador", "StandardScaler()", False)]),
    "IMPUTACION": ("con_faltantes", [("imputador", "KNNImputer(n_neighbors=5)", False)]),
    "SELECCION":  ("completo",      [("selector", "SelectKBest(score_func={F}, k=8)", True)]),
    "COMBINADA":  ("con_faltantes", [("imputador", "KNNImputer(n_neighbors=5)", False),
                                     ("escalador", "StandardScaler()", False),
                                     ("selector", "SelectKBest(score_func={F}, k=8)", True)]),
}
# LIN y MLP necesitan variables en escalas comparables. En las fugas IMPUTACION y SELECCION (donde la
# fuga NO es el escalado) se les agrega un StandardScaler DENTRO del modelo, igual en leaky y clean:
# así el escalado no es una fuente de inestabilidad ni de diferencia entre las dos versiones.
FAMILIAS_CON_ESCALADO_INTERNO = {"LIN", "MLP"}
FUGAS_CON_ESCALADO_INTERNO = {"IMPUTACION", "SELECCION"}
SALIDA_DE_PASO = {"imputador": "X_imp", "escalador": "X_esc", "selector": "X_sel"}


# %% PASO 2 · Plantilla de UN script (leaky o clean)
# ------------------------------------------------------------------------------
# QUÉ HACE   : devuelve como texto un script Python completo y ejecutable.
# LÓGICA     : construye la lista L línea a línea:
#   1) docstring con dataset, familia, fuga y versión;  2) imports mínimos;
#   3) carga de datos con cargar_dataset y N_TOTAL = len(y) (lo usa el verificador en ejecución
#      para saber cuántas filas tiene el dataset completo);  4) pliegues KFold/StratifiedKFold(5, 42);
#   5) modelo;  6a) leaky: cada transformador .fit_transform() sobre TODAS las filas y luego
#      cross_validate sobre la matriz contaminada;  6b) clean: Pipeline(transformadores + modelo)
#      y cross_validate sobre la matriz ORIGINAL.
# MÉTRICAS   : clasificación -> roc_auc_ovr y f1_macro; regresión -> r2 y neg_rmse.
# SALIDA     : el script deja en la variable global `resultados` la salida de cross_validate,
#              que run_benchmark.py lee para calcular ΔOpt.
# ------------------------------------------------------------------------------
def generar_script(id_dataset, familia, fuga, version):
    """Devuelve el texto de UN script. version: 'leaky' o 'clean'."""
    tarea = DATASETS[id_dataset]
    clf = tarea == "clasificacion"
    importacion, ctor_clf, ctor_reg = FAMILIAS[familia]
    variante, pasos = FUGAS[fuga]
    funcion_f = "f_classif" if clf else "f_regression"
    pasos = [(n, c.replace("{F}", funcion_f), usa_y) for n, c, usa_y in pasos]

    escalado_interno = familia in FAMILIAS_CON_ESCALADO_INTERNO and fuga in FUGAS_CON_ESCALADO_INTERNO
    clase_cv = "StratifiedKFold" if clf else "KFold"
    metricas = ('{"roc_auc": "roc_auc_ovr", "f1_macro": "f1_macro"}' if clf
                else '{"r2": "r2", "neg_rmse": "neg_root_mean_squared_error"}')

    L = []  # líneas del script
    L += ['"""',
          f"Pipeline CONTROLADO  |  dataset {id_dataset} ({NOMBRE_DATASET[id_dataset]}, {tarea})  |  "
          f"modelo {familia}  |  fuga {fuga}  |  versión {version.upper()}",
          "",
          "Generado automáticamente por 02_pipelines/generate_controlled_pairs.py (no editar a mano).",
          ("Esta versión TIENE fuga: los transformadores se ajustan con todas las filas antes de validar."
           if version == "leaky" else
           "Esta versión NO tiene fuga: los transformadores van dentro de un Pipeline y se ajustan por pliegue."),
          '"""',
          "import pathlib", "import sys", "",
          "from sklearn.model_selection import " + f"{clase_cv}, cross_validate"]
    clases_sk = {"escalador": "from sklearn.preprocessing import StandardScaler",
                 "imputador": "from sklearn.impute import KNNImputer",
                 "selector": f"from sklearn.feature_selection import SelectKBest, {funcion_f}"}
    for nombre, _, _ in pasos:
        L.append(clases_sk[nombre])
    if escalado_interno:
        L.append("from sklearn.preprocessing import StandardScaler")
    if version == "clean" or escalado_interno:
        L.append("from sklearn.pipeline import Pipeline")
    L += [importacion, "",
          "# Permite importar el cargador de datos que vive en 01_data/",
          "sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '01_data'))",
          "from sklearn_loader import cargar_dataset", "",
          f"# 1) Datos ({variante.replace('_', ' ')}). X = variables, y = objetivo.",
          f'X, y = cargar_dataset("{id_dataset}", variante="{variante}")',
          "N_TOTAL = len(y)  # filas del dataset completo (lo usa el verificador en ejecución)", "",
          "# 2) Validación cruzada de 5 pliegues, siempre con la misma semilla.",
          f"cv = {clase_cv}(n_splits=5, shuffle=True, random_state=42)", "",
          "# 3) Modelo (mismos hiperparámetros en la versión leaky y en la clean)."]
    estimador = ctor_clf if clf else ctor_reg
    if escalado_interno:
        L += ["#    Lleva su propio escalador interno, que se ajusta por pliegue (no es la fuga estudiada).",
              f'modelo = Pipeline([("escalador_interno", StandardScaler()), ("estimador", {estimador})])']
    else:
        L.append(f"modelo = {estimador}")
    L += ["",
          f"METRICAS = {metricas}", ""]

    if version == "leaky":
        L += ["# 4) FUGA: cada transformador se ajusta con TODAS las filas, incluidas las que luego",
              "#    serán de prueba en cada pliegue de la validación cruzada."]
        entrada = "X"
        for nombre, ctor, usa_y in pasos:
            salida = SALIDA_DE_PASO[nombre]
            L.append(f"{nombre} = {ctor}")
            L.append(f"{salida} = {nombre}.fit_transform({entrada}{', y' if usa_y else ''})"
                     f"  # <-- ajuste con todas las filas")
            entrada = salida
        L += ["", "# 5) Se evalúa con validación cruzada sobre una matriz YA contaminada.",
              f"resultados = cross_validate(modelo, {entrada}, y, cv=cv, scoring=METRICAS)"]
    else:
        L += ["# 4) CORRECTO: los transformadores y el modelo van juntos en un Pipeline.",
              "#    En cada pliegue, .fit() solo ve las filas de entrenamiento; la prueba solo se transforma.",
              "pipeline = Pipeline(["]
        for nombre, ctor, _ in pasos:
            L.append(f'    ("{nombre}", {ctor}),')
        L += ['    ("modelo", modelo),', "])", "",
              "# 5) Se evalúa con los mismos pliegues sobre la matriz ORIGINAL.",
              "resultados = cross_validate(pipeline, X, y, cv=cv, scoring=METRICAS)"]
    L.append("")
    return "\n".join(L)


# %% PASO 3 · Escritura del banco completo
# ------------------------------------------------------------------------------
# QUÉ HACE   : recorre las 6 × 4 × 4 × 2 = 192 combinaciones y escribe cada archivo con el nombre
#              <DATASET>_<FAMILIA>_<FUGA>_<version>.py. La etiqueta de verdad queda en el propio
#              nombre (_leaky / _clean), que es lo que usan run_benchmark.py y la línea base.
# SALIDA     : número de archivos escritos (debe ser 192; lo comprueba 08_tests/test_project.py).
# ------------------------------------------------------------------------------
def nombre_archivo(id_dataset, familia, fuga, version):
    return f"{id_dataset}_{familia}_{fuga}_{version}.py"


def generar_todos(carpeta=CARPETA_SALIDA):
    """Escribe los 192 scripts. Devuelve la cantidad de archivos escritos."""
    carpeta.mkdir(parents=True, exist_ok=True)
    n = 0
    for id_dataset in DATASETS:
        for familia in FAMILIAS:
            for fuga in FUGAS:
                for version in ("leaky", "clean"):
                    texto = generar_script(id_dataset, familia, fuga, version)
                    (carpeta / nombre_archivo(id_dataset, familia, fuga, version)).write_text(texto, encoding="utf-8")
                    n += 1
    return n


# %% PASO 4 · Ejecución
# ------------------------------------------------------------------------------
# Regenera los 192 scripts. Es idempotente: dos ejecuciones producen archivos idénticos.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"{generar_todos()} scripts escritos en {CARPETA_SALIDA}")
