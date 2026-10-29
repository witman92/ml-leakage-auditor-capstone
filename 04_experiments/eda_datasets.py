"""
eda_datasets.py  ·  Análisis exploratorio (EDA) de los 6 datasets  (criterio 3 de la rúbrica v2).

QUÉ RESPONDE
    ¿Qué datos usamos, qué calidad tienen, qué tan representativos son y qué limitaciones traen?
    Para cada dataset calcula: tamaño, variables, tipo de tarea, balance de clases, dispersión
    de escalas entre variables (importa para la fuga por escalado), asimetría (outliers) y
    valores faltantes de la variante "con_faltantes".

SALIDAS
    04_experiments/results/eda_datasets.csv
    04_experiments/results/fig_eda_datasets.png   (Figura: escalas y balance de clases)

    python 04_experiments/eda_datasets.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # dibuja a archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "01_data"))
from sklearn_loader import DATASETS, N_RUIDO, cargar_base, cargar_dataset  # noqa: E402

SALIDA = RAIZ / "04_experiments" / "results"


def resumen_dataset(ident):
    X, y = cargar_base(ident)
    X = np.asarray(X, float)
    nombre, tarea = DATASETS[ident]
    desviaciones = X.std(axis=0)
    desviaciones = desviaciones[desviaciones > 0]            # ignora columnas constantes (digits)
    fila = {
        "id": ident, "dataset": nombre, "tarea": tarea, "N": X.shape[0], "p_base": X.shape[1],
        "p_con_ruido": X.shape[1] + (0 if ident == "D6" else N_RUIDO),
        "columnas_constantes": int((X.std(axis=0) == 0).sum()),
        # Razón entre la variable de mayor y de menor desviación estándar: si es enorme,
        # un escalador cambia mucho los datos y su fuga puede pesar más.
        "razon_escalas_max_min": round(float(desviaciones.max() / desviaciones.min()), 1),
        "asimetria_media_abs": round(float(np.nanmean(np.abs(skew(X, axis=0, nan_policy="omit")))), 2),
    }
    if tarea == "clasificacion":
        frecuencias = pd.Series(y).value_counts(normalize=True)
        fila.update({"n_clases": len(frecuencias), "clase_minoritaria_pct": round(100 * frecuencias.min(), 1),
                     "y_media": None, "y_desv": None})
    else:
        fila.update({"n_clases": None, "clase_minoritaria_pct": None,
                     "y_media": round(float(np.mean(y)), 3), "y_desv": round(float(np.std(y)), 3)})
    Xf, _ = cargar_dataset(ident, "con_faltantes")
    fila["faltantes_variante_pct"] = round(100 * float(np.isnan(Xf).mean()), 1)
    return fila, X, y


def main():
    filas, escalas, minoritarias = [], {}, {}
    for ident in DATASETS:
        fila, X, y = resumen_dataset(ident)
        filas.append(fila)
        s = X.std(axis=0)
        escalas[f"{ident}\n{fila['dataset']}"] = np.log10(s[s > 0])
        if fila["tarea"] == "clasificacion":
            minoritarias[ident] = fila["clase_minoritaria_pct"]
    tabla = pd.DataFrame(filas)
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / "eda_datasets.csv", index=False)
    print(tabla.to_string(index=False))

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    ejes[0].boxplot(list(escalas.values()), tick_labels=list(escalas.keys()))
    ejes[0].set_ylabel("log10(desviación estándar de cada variable)")
    ejes[0].set_title("(a) Dispersión de escalas entre variables")
    ejes[1].bar(list(minoritarias), list(minoritarias.values()), color="#4C78A8")
    ejes[1].set_ylabel("% de la clase minoritaria")
    ejes[1].set_title("(b) Balance de clases (clasificación)")
    plt.tight_layout()
    fig.savefig(SALIDA / "fig_eda_datasets.png", dpi=200)
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
