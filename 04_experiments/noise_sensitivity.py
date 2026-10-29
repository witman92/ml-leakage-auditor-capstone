# ==============================================================================
# noise_sensitivity.py · Control positivo: ¿cuánto infla la selección con fuga según el ruido?
# ==============================================================================
"""
Mide el ΔOpt de la fuga por SELECCIÓN con 0, 50, 200 y 1000 columnas de ruido, en dos escenarios
("real" y "senal_nula"), para dos familias contrastantes (RandomForest y lineal).

    python 04_experiments/noise_sensitivity.py

Salidas: results/noise_sensitivity.csv   y   results/fig_noise_sensitivity.png
"""

# %% PASO 0 · Configuración del entorno e importaciones
# ------------------------------------------------------------------------------
# POR QUÉ EXISTE ESTE ANÁLISIS: en el experimento principal (50 columnas de ruido) la selección
#   con fuga casi no infla la métrica en datasets de señal fuerte. Eso NO prueba que sea inocua:
#   su daño depende de cuántas columnas inútiles hay entre las cuales "elegir"
#   (Ambroise y McLachlan, 2002). Aquí se varía el ruido de forma controlada.
# ------------------------------------------------------------------------------
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


# %% PASO 1 · Modelos comparados
# ------------------------------------------------------------------------------
# LÓGICA: dos familias contrastantes, con los mismos hiperparámetros del banco de control:
#   - RF : árboles, no necesitan escalar.
#   - LIN: regresión logística / Ridge con un escalador INTERNO (igual en leaky y clean; no es la
#          fuga estudiada).
# ------------------------------------------------------------------------------
def modelos(tarea):
    if tarea == "clasificacion":
        return {"RF": RandomForestClassifier(n_estimators=100, random_state=42),
                "LIN": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))}
    return {"RF": RandomForestRegressor(n_estimators=100, random_state=42),
            "LIN": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))}


# %% PASO 2 · Función central: ΔOpt de la selección para una celda del diseño
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: sklearn_loader.cargar_dataset(id, "completo", n_ruido) — datasets nativos
#   de scikit-learn con n_ruido columnas gaussianas añadidas (semilla fija por dataset).
# ESCENARIOS:
#   "real"       : etiquetas verdaderas.
#   "senal_nula" : etiquetas PERMUTADAS (semilla 42). Nada predice nada: la verdad es AUC = 0,5 o
#                  R² ≤ 0. Es un CONTROL POSITIVO: sin fuga el modelo debe rendir como el azar;
#                  con fuga, el selector "encuentra" columnas de ruido que parecen predecir.
# LÓGICA:
#   leaky : SelectKBest(k=8) ajustado con TODAS las filas y etiquetas, luego validación cruzada.
#   clean : Pipeline([SelectKBest(k=8), modelo]) → el selector solo ve el entrenamiento del pliegue.
# MÉTRICA: ΔOpt = métrica(leaky) − métrica(clean), mismos 5 pliegues (semilla 42).
# ------------------------------------------------------------------------------
def delta_opt(ident, n_ruido, nombre_modelo, escenario="real"):
    tarea = DATASETS[ident][1]
    X, y = cargar_dataset(ident, "completo", n_ruido=n_ruido)
    if escenario == "senal_nula":
        y = np.random.default_rng(42).permutation(y)
    clf = tarea == "clasificacion"
    cv = (StratifiedKFold if clf else KFold)(n_splits=5, shuffle=True, random_state=42)
    metrica = "roc_auc_ovr" if clf else "r2"
    funcion = f_classif if clf else f_regression
    modelo = modelos(tarea)[nombre_modelo]

    X_sel = SelectKBest(funcion, k=8).fit_transform(X, y)                       # CON FUGA
    con_fuga = cross_val_score(modelo, X_sel, y, cv=cv, scoring=metrica).mean()
    sin_fuga = cross_val_score(Pipeline([("selector", SelectKBest(funcion, k=8)), ("modelo", modelo)]),
                               X, y, cv=cv, scoring=metrica).mean()            # SIN FUGA
    return {"dataset": ident, "nombre": DATASETS[ident][0], "tarea": tarea, "familia": nombre_modelo,
            "escenario": escenario, "n_ruido": n_ruido, "metrica": metrica, "M_leaky": con_fuga, "M_clean": sin_fuga,
            "delta_opt": con_fuga - sin_fuga}


# %% PASO 3 · Gráfico: ΔOpt según el número de columnas de ruido
# ------------------------------------------------------------------------------
# QUÉ DIBUJA : fila superior = etiquetas reales; fila inferior = señal nula. Un panel por dataset,
#              una línea por familia. Eje X en escala symlog (0, 50, 200, 1000 columnas).
# CÓMO LEERLO: líneas planas arriba = el ruido no cambia nada cuando la señal es fuerte; líneas
#              crecientes abajo = la fuga fabrica rendimiento a medida que hay más ruido.
# ------------------------------------------------------------------------------
def figura(tabla, ruta):
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
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


# %% PASO 4 · Interpretación analítica automática
# ------------------------------------------------------------------------------
# CÓMO SE INTERPRETA: se compara el rango de ΔOpt entre niveles de ruido en cada escenario y se
#   localiza el máximo del control positivo. Si el máximo con señal nula es grande mientras el
#   escenario real es plano, la gravedad depende de la razón señal/ruido.
# ------------------------------------------------------------------------------
def interpretar(tabla):
    print("\nINTERPRETACIÓN")
    real = tabla[tabla["escenario"] == "real"]
    rango = real.groupby(["dataset", "familia"])["delta_opt"].agg(lambda s: s.max() - s.min()).max()
    print(f"  · Etiquetas reales: la variación máxima de ΔOpt entre niveles de ruido es {rango:.4f}; "
          "con señal fuerte el selector elige las mismas columnas informativas.")
    nula = tabla[tabla["escenario"] == "senal_nula"]
    for metrica, g in nula.groupby("metrica"):
        f = g.loc[g["delta_opt"].idxmax()]
        print(f"  · Señal nula ({metrica}): máximo ΔOpt = {f['delta_opt']:+.3f} en {f['nombre']} / {f['familia']} "
              f"con {f['n_ruido']} columnas de ruido: rendimiento ficticio fabricado por la fuga.")


# %% PASO 5 · Ejecución de punta a punta
# ------------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    filas = [delta_opt(d, n, m, e) for e in ("real", "senal_nula") for d in DATASETS_USADOS
             for n in NIVELES_RUIDO for m in ("RF", "LIN")]
    tabla = pd.DataFrame(filas).round(5)
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / "noise_sensitivity.csv", index=False)
    print(tabla.pivot_table(index=["escenario", "nombre", "familia"], columns="n_ruido",
                            values="delta_opt").round(4).to_string())
    figura(tabla, SALIDA / "fig_noise_sensitivity.png")
    interpretar(tabla)
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
