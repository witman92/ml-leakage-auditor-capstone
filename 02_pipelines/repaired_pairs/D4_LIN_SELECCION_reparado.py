"""
Pipeline CONTROLADO  |  dataset D4 (diabetes, regresion)  |  modelo LIN  |  fuga SELECCION  |  versión LEAKY

Generado automáticamente por 02_pipelines/generate_controlled_pairs.py (no editar a mano).
Esta versión TIENE fuga: los transformadores se ajustan con todas las filas antes de validar.
"""
import pathlib
import sys
from sklearn.model_selection import KFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '01_data'))
from sklearn_loader import cargar_dataset
X, y = cargar_dataset('D4', variante='completo')
N_TOTAL = len(y)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
modelo = Pipeline([('escalador_interno', StandardScaler()), ('estimador', Ridge(alpha=1.0, random_state=42))])
METRICAS = {'r2': 'r2', 'neg_rmse': 'neg_root_mean_squared_error'}
resultados = cross_validate(Pipeline([('selector', SelectKBest(score_func=f_regression, k=8)), ('modelo', modelo)]), X, y, cv=cv, scoring=METRICAS)
