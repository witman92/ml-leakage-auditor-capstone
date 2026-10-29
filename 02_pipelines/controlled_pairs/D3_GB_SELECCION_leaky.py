"""
Pipeline CONTROLADO  |  dataset D3 (digits, clasificacion)  |  modelo GB  |  fuga SELECCION  |  versión LEAKY

Generado automáticamente por 02_pipelines/generar_pares_controlados.py (no editar a mano).
Esta versión TIENE fuga: los transformadores se ajustan con todas las filas antes de validar.
"""
import pathlib
import sys

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# Permite importar el cargador de datos que vive en 01_data/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '01_data'))
from sklearn_loader import cargar_dataset

# 1) Datos (completo). X = variables, y = objetivo.
X, y = cargar_dataset("D3", variante="completo")
N_TOTAL = len(y)  # filas del dataset completo (lo usa el verificador en ejecución)

# 2) Validación cruzada de 5 pliegues, siempre con la misma semilla.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3) Modelo (mismos hiperparámetros en la versión leaky y en la clean).
modelo = HistGradientBoostingClassifier(random_state=42)

METRICAS = {"roc_auc": "roc_auc_ovr", "f1_macro": "f1_macro"}

# 4) FUGA: cada transformador se ajusta con TODAS las filas, incluidas las que luego
#    serán de prueba en cada pliegue de la validación cruzada.
selector = SelectKBest(score_func=f_classif, k=8)
X_sel = selector.fit_transform(X, y)  # <-- ajuste con todas las filas

# 5) Se evalúa con validación cruzada sobre una matriz YA contaminada.
resultados = cross_validate(modelo, X_sel, y, cv=cv, scoring=METRICAS)
