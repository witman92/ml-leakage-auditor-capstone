"""
Pipeline CONTROLADO  |  dataset D1 (breast_cancer, clasificacion)  |  modelo MLP  |  fuga ESCALADO  |  versión CLEAN

Generado automáticamente por 02_pipelines/generate_controlled_pairs.py (no editar a mano).
Esta versión NO tiene fuga: los transformadores van dentro de un Pipeline y se ajustan por pliegue.
"""
import pathlib
import sys

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Permite importar el cargador de datos que vive en 01_data/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '01_data'))
from sklearn_loader import cargar_dataset

# 1) Datos (completo). X = variables, y = objetivo.
X, y = cargar_dataset("D1", variante="completo")
N_TOTAL = len(y)  # filas del dataset completo (lo usa el verificador en ejecución)

# 2) Validación cruzada de 5 pliegues, siempre con la misma semilla.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3) Modelo (mismos hiperparámetros en la versión leaky y en la clean).
modelo = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

METRICAS = {"roc_auc": "roc_auc_ovr", "f1_macro": "f1_macro"}

# 4) CORRECTO: los transformadores y el modelo van juntos en un Pipeline.
#    En cada pliegue, .fit() solo ve las filas de entrenamiento; la prueba solo se transforma.
pipeline = Pipeline([
    ("escalador", StandardScaler()),
    ("modelo", modelo),
])

# 5) Se evalúa con los mismos pliegues sobre la matriz ORIGINAL.
resultados = cross_validate(pipeline, X, y, cv=cv, scoring=METRICAS)
