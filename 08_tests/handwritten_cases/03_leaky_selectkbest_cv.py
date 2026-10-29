"""Fuga (R-CV): SelectKBest elige 5 variables mirando las etiquetas y de TODAS las filas y la
matriz reducida entra a cross_val_score; cada pliegue de prueba influyó en la selección.
Etiqueta real: fuga. AST: debe detectarla. Reparable automáticamente (patrón CV)."""
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

sel = SelectKBest(f_classif, k=5)
X_sel = sel.fit_transform(X, y)
print(cross_val_score(modelo, X_sel, y, cv=5).mean())
