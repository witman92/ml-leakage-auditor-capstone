# ==============================================================================
# safe_sklearn_template.py · Plantilla de pipeline SIN fuga de datos (Skill create-leakage-free-pipeline)
# ==============================================================================
"""

Cómo usarla: copie este archivo, cambie la sección "1) DATOS" por sus datos y, si hace falta,
el modelo en "3) MODELO". NO cambie el orden de los pasos: es lo que evita la fuga.

REGLAS QUE CUMPLE (y que el auditor verifica)
  R1. Ningún .fit()/.fit_transform() se llama sobre X completo: todo preprocesamiento vive
      dentro de un sklearn.pipeline.Pipeline.
  R2. Se aparta un conjunto de PRUEBA FINAL antes de hacer cualquier otra cosa y solo se
      usa UNA vez, al final.
  R3. La búsqueda de hiperparámetros (GridSearchCV) recibe el Pipeline completo, de modo que
      en cada pliegue el preprocesamiento se re-ajusta solo con las filas de entrenamiento.
  R4. Todas las semillas están fijas (SEMILLA) para que el resultado sea reproducible.
"""

# %% PASO 0 · Importaciones y configuración
# ------------------------------------------------------------------------------
# QUÉ HACE   : importa scikit-learn y fija la semilla (R4) y la tarea.
# ------------------------------------------------------------------------------
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEMILLA = 42
TAREA = "clasificacion"          # "clasificacion" o "regresion"

# %% PASO 1 · DATOS (reemplazar por los suyos)
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: de ejemplo, load_breast_cancer de scikit-learn como DataFrame (569 × 30).
# QUÉ CAMBIAR: solo esta celda (y la tarea en el PASO 0). X puede ser DataFrame o arreglo NumPy.
# ------------------------------------------------------------------------------
# X puede ser un DataFrame de pandas (con columnas numéricas y de texto) o un arreglo NumPy.
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# %% PASO 2 · PRUEBA FINAL APARTE (R2)
# ------------------------------------------------------------------------------
# LÓGICA     : la partición 80/20 se hace ANTES de cualquier ajuste (estratificada en clasificación).
# POR QUÉ    : la prueba final estima el rendimiento en datos nunca vistos; si algo se ajustara con
#              ella, la estimación sería optimista (fuga).
# ------------------------------------------------------------------------------
# Esta partición se hace PRIMERO. X_prueba/y_prueba no se tocan hasta el paso 6.
estratificar = y if TAREA == "clasificacion" else None
X_entren, X_prueba, y_entren, y_prueba = train_test_split(
    X, y, test_size=0.2, random_state=SEMILLA, stratify=estratificar)

# %% PASO 3 · MODELO, MÉTRICA Y PLIEGUES
# ------------------------------------------------------------------------------
# LÓGICA     : clasificación -> RandomForestClassifier, ROC-AUC, StratifiedKFold(5);
#              regresión     -> RandomForestRegressor, R², KFold(5). Semilla fija en todo.
# ------------------------------------------------------------------------------
if TAREA == "clasificacion":
    modelo = RandomForestClassifier(random_state=SEMILLA)
    funcion_seleccion, metrica_cv, cv = f_classif, "roc_auc", StratifiedKFold(5, shuffle=True, random_state=SEMILLA)
else:
    modelo = RandomForestRegressor(random_state=SEMILLA)
    funcion_seleccion, metrica_cv, cv = f_regression, "r2", KFold(5, shuffle=True, random_state=SEMILLA)

# %% PASO 4 · PREPROCESAMIENTO DENTRO DEL PIPELINE (R1)
# ------------------------------------------------------------------------------
# LÓGICA     : ColumnTransformer separa columnas numéricas (mediana + escalado) y de texto (moda +
#              one-hot). Todo, incluido SelectKBest (que usa y), vive dentro de un único Pipeline,
#              de modo que cada .fit() solo ve las filas de entrenamiento que le toquen.
# ------------------------------------------------------------------------------
# Columnas numéricas: imputar con la mediana y escalar. Columnas de texto: imputar y codificar.
# Si X es un arreglo NumPy sin columnas de texto, el bloque 'categoricas' queda vacío.
numericas = X_entren.select_dtypes(include="number").columns.tolist()
categoricas = [c for c in X_entren.columns if c not in numericas]
preprocesamiento = ColumnTransformer([
    ("num", Pipeline([("imputador", SimpleImputer(strategy="median")),
                      ("escalador", StandardScaler())]), numericas),
    ("cat", Pipeline([("imputador", SimpleImputer(strategy="most_frequent")),
                      ("codificador", OneHotEncoder(handle_unknown="ignore"))]), categoricas),
])
pipeline = Pipeline([
    ("preprocesamiento", preprocesamiento),
    ("selector", SelectKBest(score_func=funcion_seleccion)),   # también DENTRO: usa y, y es fuga si va fuera
    ("modelo", modelo),
])

# %% PASO 5 · AJUSTE DE HIPERPARÁMETROS (R3)
# ------------------------------------------------------------------------------
# LÓGICA     : GridSearchCV recibe el Pipeline COMPLETO y la matriz de entrenamiento original.
# MÉTRICA    : best_score_ es el promedio de validación cruzada interna; sirve para elegir, no
#              para reportar el rendimiento final (eso lo hace el PASO 6).
# ------------------------------------------------------------------------------
# Se le pasa el pipeline COMPLETO: en cada pliegue, imputador, escalador y selector se re-ajustan
# solo con las filas de entrenamiento de ese pliegue.
rejilla = {"selector__k": [5, 10, "all"], "modelo__n_estimators": [100, 200]}
busqueda = GridSearchCV(pipeline, rejilla, cv=cv, scoring=metrica_cv)
busqueda.fit(X_entren, y_entren)
print(f"Mejores parámetros: {busqueda.best_params_}")
print(f"Puntaje en validación cruzada ({metrica_cv}): {busqueda.best_score_:.4f}")

# %% PASO 6 · EVALUACIÓN FINAL, UNA SOLA VEZ
# ------------------------------------------------------------------------------
# INTERPRETACIÓN: esta cifra es la estimación honesta del rendimiento. Si difiere mucho de la de
#              validación cruzada, revise el tamaño de la muestra o la estabilidad del modelo.
# ------------------------------------------------------------------------------
if TAREA == "clasificacion":
    puntaje = roc_auc_score(y_prueba, busqueda.predict_proba(X_prueba)[:, 1])
    print(f"ROC-AUC en la prueba final: {puntaje:.4f}")
else:
    prediccion = busqueda.predict(X_prueba)
    print(f"R2 en la prueba final: {r2_score(y_prueba, prediccion):.4f} | "
          f"RMSE: {np.sqrt(mean_squared_error(y_prueba, prediccion)):.4f}")
