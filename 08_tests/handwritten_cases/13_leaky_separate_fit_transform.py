"""Fuga (R-Split): selector.fit(X, y) y selector.transform(X) en líneas separadas, antes de partir;
la selección usó las etiquetas de las filas de prueba. Etiqueta real: fuga. AST: debe detectarla."""
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
sel.fit(X, y)
X_new = sel.transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_new, y, test_size=0.3, random_state=0)
print(modelo.fit(X_tr, y_tr).score(X_te, y_te))
