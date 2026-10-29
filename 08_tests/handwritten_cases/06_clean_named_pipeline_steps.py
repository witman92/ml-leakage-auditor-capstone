"""Limpio: Pipeline con pasos nombrados (imputador, escalador, modelo) dentro de cross_val_score.
Etiqueta real: sin fuga."""
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

pipe = Pipeline([('imp', SimpleImputer()), ('esc', StandardScaler()), ('m', modelo)])
print(cross_val_score(pipe, X, y, cv=5).mean())
