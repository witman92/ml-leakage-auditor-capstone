"""
Pipeline CONTROLADO  |  dataset D5 (california_housing, regresion)  |  modelo MLP  |  fuga SELECCION  |  versión LEAKY

Generado automáticamente por 02_pipelines/generar_pares_controlados.py (no editar a mano).
Esta versión TIENE fuga: los transformadores se ajustan con todas las filas antes de validar.
"""
import pathlib
import sys

from sklearn.model_selection import KFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Permite importar el cargador de datos que vive en 01_data/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '01_data'))
from sklearn_loader import cargar_dataset

# 1) Datos (completo). X = variables, y = objetivo.
X, y = cargar_dataset("D5", variante="completo")
N_TOTAL = len(y)  # filas del dataset completo (lo usa el verificador en ejecución)

# 2) Validación cruzada de 5 pliegues, siempre con la misma semilla.
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 3) Modelo (mismos hiperparámetros en la versión leaky y en la clean).
#    Lleva su propio escalador interno, que se ajusta por pliegue (no es la fuga estudiada).
modelo = Pipeline([("escalador_interno", StandardScaler()), ("estimador", MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))])

METRICAS = {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error"}

# 4) FUGA: cada transformador se ajusta con TODAS las filas, incluidas las que luego
#    serán de prueba en cada pliegue de la validación cruzada.
selector = SelectKBest(score_func=f_regression, k=8)
X_sel = selector.fit_transform(X, y)  # <-- ajuste con todas las filas

# 5) Se evalúa con validación cruzada sobre una matriz YA contaminada.
resultados = cross_validate(modelo, X_sel, y, cv=cv, scoring=METRICAS)
