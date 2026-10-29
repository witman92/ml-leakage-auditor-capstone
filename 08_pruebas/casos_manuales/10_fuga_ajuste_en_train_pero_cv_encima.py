"""Fuga leve: escalador ajustado con X_train y luego CV dentro de X_train."""
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
X_tr_s = scaler.fit_transform(X_tr)
print(cross_val_score(modelo, X_tr_s, y_tr, cv=5).mean())
