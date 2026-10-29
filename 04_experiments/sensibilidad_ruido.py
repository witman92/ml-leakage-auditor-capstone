"""
sensibilidad_ruido.py  ·  ¿Cuánta fuga produce la selección de variables según cuánto ruido hay?

POR QUÉ EXISTE ESTE ANÁLISIS
    En el experimento principal se agregan 50 columnas de ruido. En datasets con señal muy
    fuerte (breast_cancer, digits) el selector, con o sin fuga, elige las mismas columnas
    informativas y el ΔOpt sale casi cero. Eso NO prueba que la fuga sea inofensiva: el daño
    de seleccionar variables con las etiquetas de todo el dataset depende de cuántas columnas
    inútiles hay entre las cuales "elegir" (Ambroise & McLachlan, 2002).
    Este script lo mide: repite SOLO la fuga por SELECCIÓN con 0, 50, 200 y 1000 columnas de ruido,
    en dos escenarios:
      "real"        : las etiquetas verdaderas (señal fuerte). Aquí el selector nunca elige columnas de
                      ruido, así que el nivel de ruido no cambia el resultado.
      "senal_nula"  : las etiquetas se PERMUTAN al azar (la verdad es que NADA predice nada: AUC = 0,5
                      o R² <= 0). Es un CONTROL POSITIVO: sin fuga el modelo debe rendir como el azar;
                      con fuga, el selector "encuentra" columnas de ruido que parecen predecir y el
                      puntaje sube. Muestra que ΔOpt detecta la fuga cuando la fuga importa.

QUÉ COMPARA (en cada celda)
    leaky : SelectKBest(k=8) ajustado con TODAS las filas y etiquetas, luego validación cruzada.
    clean : Pipeline([SelectKBest(k=8), modelo]) -> el selector solo ve el entrenamiento de cada pliegue.
    ΔOpt  = métrica(leaky) − métrica(clean), con los mismos pliegues (semilla 42).

SALIDAS   results/sensibilidad_ruido.csv   y   results/fig_sensibilidad_ruido.png

    python 04_experiments/sensibilidad_ruido.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "01_data"))
from sklearn_loader import DATASETS, cargar_dataset  # noqa: E402

SALIDA = RAIZ / "04_experiments" / "results"
NIVELES_RUIDO = [0, 50, 200, 1000]          # columnas de ruido añadidas
DATASETS_USADOS = ["D1", "D2", "D4"]        # 2 de clasificación (AUC) y 1 de regresión (R²)


def modelos(tarea):
    """Dos familias contrastantes: árboles (no necesitan escalar) y lineal (con escalador interno)."""
    if tarea == "clasificacion":
        return {"RF": RandomForestClassifier(n_estimators=100, random_state=42),
                "LIN": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))}
    return {"RF": RandomForestRegressor(n_estimators=100, random_state=42),
            "LIN": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))}


def delta_opt(ident, n_ruido, nombre_modelo, escenario="real"):
    """ΔOpt de la fuga por selección para un dataset, un nivel de ruido, una familia y un escenario."""
    tarea = DATASETS[ident][1]
    X, y = cargar_dataset(ident, "completo", n_ruido=n_ruido)
    if escenario == "senal_nula":                 # control positivo: se rompe la relación X -> y
        y = np.random.default_rng(42).permutation(y)
    clf = tarea == "clasificacion"
    cv = (StratifiedKFold if clf else KFold)(n_splits=5, shuffle=True, random_state=42)
    metrica = "roc_auc_ovr" if clf else "r2"
    funcion = f_classif if clf else f_regression
    modelo = modelos(tarea)[nombre_modelo]

    # CON FUGA: el selector mira todas las filas y todas las etiquetas antes de la validación.
    X_sel = SelectKBest(funcion, k=8).fit_transform(X, y)
    con_fuga = cross_val_score(modelo, X_sel, y, cv=cv, scoring=metrica).mean()
    # SIN FUGA: el selector va dentro del Pipeline y se re-ajusta en cada pliegue.
    sin_fuga = cross_val_score(Pipeline([("selector", SelectKBest(funcion, k=8)), ("modelo", modelo)]),
                               X, y, cv=cv, scoring=metrica).mean()
    return {"dataset": ident, "nombre": DATASETS[ident][0], "tarea": tarea, "familia": nombre_modelo,
            "escenario": escenario, "n_ruido": n_ruido, "metrica": metrica, "M_leaky": con_fuga, "M_clean": sin_fuga,
            "delta_opt": con_fuga - sin_fuga}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    filas = [delta_opt(d, n, m, e) for e in ("real", "senal_nula") for d in DATASETS_USADOS
             for n in NIVELES_RUIDO for m in ("RF", "LIN")]
    tabla = pd.DataFrame(filas).round(5)
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / "sensibilidad_ruido.csv", index=False)
    print(tabla.pivot_table(index=["escenario", "nombre", "familia"], columns="n_ruido",
                            values="delta_opt").round(4).to_string())

    fig, ejes = plt.subplots(2, len(DATASETS_USADOS), figsize=(11, 6), sharex=True)
    for fila, escenario in enumerate(("real", "senal_nula")):
        for eje, ident in zip(ejes[fila], DATASETS_USADOS):
            sub = tabla[(tabla["dataset"] == ident) & (tabla["escenario"] == escenario)]
            for fam in ("RF", "LIN"):
                s = sub[sub["familia"] == fam]
                eje.plot(s["n_ruido"], s["delta_opt"], marker="o", label=fam)
            eje.axhline(0, color="black", lw=0.8)
            eje.set_xscale("symlog", linthresh=10)
            eje.set_title(f"{ident} {DATASETS[ident][0]} · {escenario} ({sub['metrica'].iloc[0]})", fontsize=9)
        ejes[fila][0].set_ylabel("ΔOpt (fuga por selección)")
    for eje in ejes[1]:
        eje.set_xlabel("columnas de ruido")
    ejes[0][0].legend()
    plt.tight_layout()
    fig.savefig(SALIDA / "fig_sensibilidad_ruido.png", dpi=200)
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
