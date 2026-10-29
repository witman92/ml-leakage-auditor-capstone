"""
Pipeline CONTROLADO  |  dataset D5 (california_housing, regresion)  |  modelo RF  |  fuga COMBINADA  |  versión LEAKY

Generado automáticamente por 02_pipelines/generar_pares_controlados.py (no editar a mano).
Esta versión TIENE fuga: los transformadores se ajustan con todas las filas antes de validar.
"""
import pathlib
import sys
from sklearn.model_selection import KFold, cross_validate
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '01_data'))
from sklearn_loader import cargar_dataset
from sklearn.pipeline import Pipeline
X, y = cargar_dataset('D5', variante='con_faltantes')
N_TOTAL = len(y)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
METRICAS = {'r2': 'r2', 'neg_rmse': 'neg_root_mean_squared_error'}
resultados = cross_validate(Pipeline([('imputador', KNNImputer(n_neighbors=5)), ('escalador', StandardScaler()), ('selector', SelectKBest(score_func=f_regression, k=8)), ('modelo', modelo)]), X, y, cv=cv, scoring=METRICAS)
