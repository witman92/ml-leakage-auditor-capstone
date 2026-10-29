"""Fuga con alias de importación: 'StandardScaler as SS' ajustado con X completo antes de la
validación cruzada. LIMITACIÓN CONOCIDA del AST (sigue nombres de clase, no alias): no la detecta.
El verificador en ejecución sí la ve (el escalador recibe el 100 % de las filas). Etiqueta: fuga."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline

X, y = load_wine(return_X_y=True)
modelo = LogisticRegression(max_iter=2000)

from sklearn.preprocessing import StandardScaler as SS
X_p = SS().fit_transform(X)
print(cross_val_score(modelo, X_p, y, cv=5).mean())
