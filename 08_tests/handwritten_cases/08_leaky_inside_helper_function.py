"""Fuga (R-CV, confianza media): el escalado global está escondido en la función auxiliar
preparar(), que se aplica a X completo antes de cross_val_score. El AST no sigue el flujo dentro
de la función, pero lo marca como sospecha. Etiqueta real: fuga. No reparable automáticamente."""
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

def preparar(M):
    return StandardScaler().fit_transform(M)

X_p = preparar(X)
print(cross_val_score(modelo, X_p, y, cv=5).mean())
