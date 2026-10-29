"""Fuga (R-Split): el escalador se ajusta con las 178 filas y DESPUÉS se parte en train/test.
La media y la desviación del StandardScaler incluyen las filas de prueba, que luego se usan para
evaluar. Etiqueta real: fuga. AST: debe detectarla. Reparable automáticamente (patrón Split)."""
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

scaler = StandardScaler()
X_esc = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_esc, y, test_size=0.3, random_state=0)
modelo.fit(X_tr, y_tr)
print(modelo.score(X_te, y_te))
