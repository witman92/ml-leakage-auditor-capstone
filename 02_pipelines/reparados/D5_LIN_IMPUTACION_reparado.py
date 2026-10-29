"""
Pipeline CONTROLADO  |  dataset D5 (california_housing, regresion)  |  modelo LIN  |  fuga IMPUTACION  |  versión LEAKY

Generado automáticamente por 02_pipelines/generar_pares_controlados.py (no editar a mano).
Esta versión TIENE fuga: los transformadores se ajustan con todas las filas antes de validar.
"""
import pathlib
import sys
from sklearn.model_selection import KFold, cross_validate
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '01_data'))
from sklearn_loader import cargar_dataset
X, y = cargar_dataset('D5', variante='con_faltantes')
N_TOTAL = len(y)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
modelo = Pipeline([('escalador_interno', StandardScaler()), ('estimador', Ridge(alpha=1.0, random_state=42))])
METRICAS = {'r2': 'r2', 'neg_rmse': 'neg_root_mean_squared_error'}
resultados = cross_validate(Pipeline([('imputador', KNNImputer(n_neighbors=5)), ('modelo', modelo)]), X, y, cv=cv, scoring=METRICAS)
