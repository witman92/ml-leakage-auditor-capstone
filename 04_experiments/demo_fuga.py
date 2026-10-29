"""
demo_fuga.py  ·  Ejemplo mínimo y visual de fuga de datos (observación R3 del comité).

QUÉ MUESTRA
    El MISMO modelo, los MISMOS datos y los MISMOS pliegues evaluados de dos maneras:
      A) CON FUGA : imputador, escalador y selector se ajustan con TODAS las filas antes de la
                    validación cruzada (patrón típico de código generado por LLM).
      B) SIN FUGA : los tres pasos van dentro de un Pipeline; en cada pliegue, .fit() solo ve
                    el 80 % de entrenamiento.
    La diferencia de ROC-AUC entre A y B es el Sesgo de Optimismo  ΔOpt = M_leaky − M_clean.

QUÉ ESPERAR (honestidad)
    Con breast_cancer (señal fuerte, AUC ≈ 0,99) el ΔOpt es muy pequeño: efecto techo. La fuga
    SÍ existe, pero casi no se ve. El segundo experimento (señal nula) hace visible el mecanismo:
    con etiquetas al azar la verdad es AUC = 0,5, y la selección con fuga "encuentra" columnas
    que por azar parecen predecir.

SALIDAS
    04_experiments/results/demo_fuga.csv
    04_experiments/results/fig_demo_fuga.png

    python 04_experiments/demo_fuga.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # dibuja a archivo; no depende de una ventana
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")           # avisos de convergencia del MLP: no afectan la lección

SEMILLA = 42
SALIDA = Path(__file__).resolve().parent / "results"


def modelo():
    return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEMILLA)


def comparar(X_faltante, y, k, etiqueta):
    """Devuelve (scores_leaky, scores_clean) con los mismos pliegues."""
    pliegues = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    # --- A) CON FUGA: los tres .fit() ocurren antes de dividir en pliegues ---
    X_imp = KNNImputer(n_neighbors=5).fit_transform(X_faltante)              # L2: imputación
    X_esc = StandardScaler().fit_transform(X_imp)                            # L1: escalado
    X_sel = SelectKBest(f_classif, k=k).fit_transform(X_esc, y)              # L3: selección (usa y)
    leaky = cross_val_score(modelo(), X_sel, y, cv=pliegues, scoring="roc_auc")

    # --- B) SIN FUGA: todo dentro del Pipeline; se reajusta en cada pliegue ---
    limpio = Pipeline([("imputador", KNNImputer(n_neighbors=5)),
                       ("escalador", StandardScaler()),
                       ("selector", SelectKBest(f_classif, k=k)),
                       ("modelo", modelo())])
    clean = cross_val_score(limpio, X_faltante, y, cv=pliegues, scoring="roc_auc")
    print(f"[{etiqueta}] con fuga {leaky.mean():.4f} | sin fuga {clean.mean():.4f} | "
          f"ΔOpt = {leaky.mean() - clean.mean():+.4f}")
    return leaky, clean


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("Cargando breast_cancer (nativo de scikit-learn)...")
    X, y = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(SEMILLA)

    # 10 % de celdas vacías, determinista, para que el imputador tenga algo que hacer
    X_faltante = X.copy()
    X_faltante[rng.random(X.shape) < 0.10] = np.nan

    # Escenario 1: datos reales (señal fuerte)
    real = comparar(X_faltante, y, k=8, etiqueta="señal real      ")

    # Escenario 2: señal nula = 500 columnas de ruido + etiquetas permutadas (verdad: AUC = 0,5)
    y_azar = rng.permutation(y)
    X_ruido = np.hstack([X_faltante, rng.normal(size=(X.shape[0], 500))])
    nula = comparar(X_ruido, y_azar, k=8, etiqueta="señal nula      ")

    filas = []
    for escenario, (leaky, clean) in (("señal real", real), ("señal nula (verdad AUC=0,5)", nula)):
        filas.append({"escenario": escenario, "auc_con_fuga": round(leaky.mean(), 4),
                      "auc_sin_fuga": round(clean.mean(), 4), "delta_opt": round(leaky.mean() - clean.mean(), 4),
                      "sd_con_fuga": round(leaky.std(), 4), "sd_sin_fuga": round(clean.std(), 4)})
    tabla = pd.DataFrame(filas)
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / "demo_fuga.csv", index=False)

    # --- Figura: dos paneles, uno por escenario ---
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for eje, (titulo, (leaky, clean)) in zip(ejes, (("(a) Señal real", real),
                                                    ("(b) Señal nula (verdad: AUC = 0,5)", nula))):
        barras = eje.bar(["Con fuga", "Sin fuga"], [leaky.mean(), clean.mean()],
                         yerr=[leaky.std(), clean.std()], capsize=8,
                         color=["#d9534f", "#4C78A8"], edgecolor="black")
        for barra in barras:
            eje.text(barra.get_x() + barra.get_width() / 2, 0.02, f"{barra.get_height():.3f}",
                     ha="center", color="white", fontweight="bold")
        eje.axhline(0.5, color="gray", ls="--", lw=1)
        eje.set_title(titulo)
        eje.set_ylim(0, 1.05)
    ejes[0].set_ylabel("ROC-AUC (5 pliegues)")
    fig.suptitle(r"Sesgo de optimismo ($\Delta$Opt) del mismo pipeline con y sin fuga")
    plt.tight_layout()
    fig.savefig(SALIDA / "fig_demo_fuga.png", dpi=200)
    print(tabla.to_string(index=False))
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
