"""Fuga (R-Test): tras partir, el escalador se re-ajusta con los datos de prueba (variable X_te),
de modo que la prueba se normaliza con sus propias estadísticas. Etiqueta real: fuga. AST: debe
detectarla por el nombre X_te; el verificador en ejecución no (no usa todas las filas)."""
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

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
scaler = StandardScaler()
scaler.fit(X_tr)
X_te_esc = scaler.fit_transform(X_te)
print(modelo.fit(scaler.transform(X_tr), y_tr).score(X_te_esc, y_te))
