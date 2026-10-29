"""
Pipeline CONTROLADO  |  dataset D2 (wine, clasificacion)  |  modelo LIN  |  fuga SELECCION  |  versión CLEAN

Generado automáticamente por 02_pipelines/generar_pares_controlados.py (no editar a mano).
Esta versión NO tiene fuga: los transformadores van dentro de un Pipeline y se ajustan por pliegue.
"""
import pathlib
import sys

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge

# Permite importar el cargador de datos que vive en 01_data/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '01_data'))
from sklearn_loader import cargar_dataset

# 1) Datos (completo). X = variables, y = objetivo.
X, y = cargar_dataset("D2", variante="completo")
N_TOTAL = len(y)  # filas del dataset completo (lo usa el verificador en ejecución)

# 2) Validación cruzada de 5 pliegues, siempre con la misma semilla.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3) Modelo (mismos hiperparámetros en la versión leaky y en la clean).
#    Lleva su propio escalador interno, que se ajusta por pliegue (no es la fuga estudiada).
modelo = Pipeline([("escalador_interno", StandardScaler()), ("estimador", LogisticRegression(C=1.0, max_iter=2000, random_state=42))])

METRICAS = {"roc_auc": "roc_auc_ovr", "f1_macro": "f1_macro"}

# 4) CORRECTO: los transformadores y el modelo van juntos en un Pipeline.
#    En cada pliegue, .fit() solo ve las filas de entrenamiento; la prueba solo se transforma.
pipeline = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=8)),
    ("modelo", modelo),
])

# 5) Se evalúa con los mismos pliegues sobre la matriz ORIGINAL.
resultados = cross_validate(pipeline, X, y, cv=cv, scoring=METRICAS)
