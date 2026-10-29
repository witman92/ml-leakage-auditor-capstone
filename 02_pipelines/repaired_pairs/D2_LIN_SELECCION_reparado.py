"""
Pipeline CONTROLADO  |  dataset D2 (wine, clasificacion)  |  modelo LIN  |  fuga SELECCION  |  versión LEAKY

Generado automáticamente por 02_pipelines/generate_controlled_pairs.py (no editar a mano).
Esta versión TIENE fuga: los transformadores se ajustan con todas las filas antes de validar.
"""
import pathlib
import sys
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '01_data'))
from sklearn_loader import cargar_dataset
X, y = cargar_dataset('D2', variante='completo')
N_TOTAL = len(y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
modelo = Pipeline([('escalador_interno', StandardScaler()), ('estimador', LogisticRegression(C=1.0, max_iter=2000, random_state=42))])
METRICAS = {'roc_auc': 'roc_auc_ovr', 'f1_macro': 'f1_macro'}
resultados = cross_validate(Pipeline([('selector', SelectKBest(score_func=f_classif, k=8)), ('modelo', modelo)]), X, y, cv=cv, scoring=METRICAS)
