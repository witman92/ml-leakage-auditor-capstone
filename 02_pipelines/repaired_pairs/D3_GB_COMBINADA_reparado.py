"""
Pipeline CONTROLADO  |  dataset D3 (digits, clasificacion)  |  modelo GB  |  fuga COMBINADA  |  versión LEAKY

Generado automáticamente por 02_pipelines/generate_controlled_pairs.py (no editar a mano).
Esta versión TIENE fuga: los transformadores se ajustan con todas las filas antes de validar.
"""
import pathlib
import sys
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '01_data'))
from sklearn_loader import cargar_dataset
from sklearn.pipeline import Pipeline
X, y = cargar_dataset('D3', variante='con_faltantes')
N_TOTAL = len(y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
modelo = HistGradientBoostingClassifier(random_state=42)
METRICAS = {'roc_auc': 'roc_auc_ovr', 'f1_macro': 'f1_macro'}
resultados = cross_validate(Pipeline([('imputador', KNNImputer(n_neighbors=5)), ('escalador', StandardScaler()), ('selector', SelectKBest(score_func=f_classif, k=8)), ('modelo', modelo)]), X, y, cv=cv, scoring=METRICAS)
