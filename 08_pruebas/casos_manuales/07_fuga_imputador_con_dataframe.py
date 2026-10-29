"""Fuga: imputación sobre DataFrame antes de la validación cruzada."""
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

df = pd.DataFrame(X)
df = df.mask(np.random.default_rng(0).random(df.shape) < 0.1)
imp = SimpleImputer(strategy='mean')
X_imp = pd.DataFrame(imp.fit_transform(df))
print(cross_val_score(modelo, X_imp, y, cv=5).mean())
