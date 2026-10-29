# ==============================================================================
# eda_datasets.py · Análisis exploratorio (EDA) de los 6 datasets del estudio
# ==============================================================================
"""
Caracterización, calidad, representatividad y limitaciones de los datos (criterio 3 de la rúbrica).

    python 04_experiments/eda_datasets.py

Salidas: results/eda_datasets.csv   y   results/fig_eda_datasets.png
"""

# %% PASO 0 · Configuración del entorno e importaciones
# ------------------------------------------------------------------------------
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


# %% PASO 1 · Resumen de UN dataset
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: sklearn_loader.cargar_base(id) → datos ORIGINALES de scikit-learn (sin ruido
#   ni faltantes); y cargar_dataset(id, "con_faltantes") para medir la variante con 10 % de NaN.
# MÉTRICAS QUE CALCULA Y POR QUÉ IMPORTAN PARA LA FUGA:
#   - N, p (variables base y con ruido), columnas constantes.
#   - Razón de escalas = desviación máx / mín entre variables: si es enorme, un escalador cambia
#     mucho los datos y su fuga podría pesar más en modelos sensibles a la escala.
#   - Asimetría media |skew|: indica atípicos que pueden desestabilizar a la red neuronal.
#   - Filas duplicadas y % de celdas atípicas (|z| > 3): calidad más allá de los faltantes.
#   - Balance de clases (clasificación) o media/desviación de y (regresión).
# ------------------------------------------------------------------------------
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
        "razon_escalas_max_min": round(float(desviaciones.max() / desviaciones.min()), 1),
        "asimetria_media_abs": round(float(np.nanmean(np.abs(skew(X, axis=0, nan_policy="omit")))), 2),
    }
    fila["filas_duplicadas"] = int(pd.DataFrame(X).duplicated().sum())
    z = np.abs((X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, np.nan, X.std(axis=0)))
    fila["atipicos_pct"] = round(100 * float(np.nanmean(z > 3)), 2)
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


# %% PASO 2 · Gráfico: dispersión de escalas y balance de clases
# ------------------------------------------------------------------------------
# QUÉ DIBUJA : (a) diagrama de caja del log10 de la desviación estándar de cada variable, por
#              dataset; (b) porcentaje de la clase minoritaria en los 3 datasets de clasificación.
# CÓMO LEERLO: una caja alta en (a) = variables en escalas muy distintas (el escalado importa);
#              barras bajas en (b) = desbalance, que justifica la validación estratificada.
# ------------------------------------------------------------------------------
def figura(escalas, minoritarias, ruta):
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    ejes[0].boxplot(list(escalas.values()), tick_labels=list(escalas.keys()))
    ejes[0].set_ylabel("log10(desviación estándar de cada variable)")
    ejes[0].set_title("(a) Dispersión de escalas entre variables")
    ejes[1].bar(list(minoritarias), list(minoritarias.values()), color="#4C78A8")
    ejes[1].set_ylabel("% de la clase minoritaria")
    ejes[1].set_title("(b) Balance de clases (clasificación)")
    plt.tight_layout()
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


# %% PASO 3 · Interpretación analítica automática
# ------------------------------------------------------------------------------
# CÓMO SE INTERPRETA: datos sin duplicados y con pocos atípicos son "limpios por diseño"; por eso
#   los faltantes se simulan. La dispersión de escalas y la asimetría anticipan dónde la fuga por
#   escalado y la inestabilidad de la MLP pueden aparecer.
# ------------------------------------------------------------------------------
def interpretar(tabla):
    print("\nINTERPRETACIÓN")
    print(f"  · Filas duplicadas en total: {int(tabla['filas_duplicadas'].sum())}; celdas atípicas entre "
          f"{tabla['atipicos_pct'].min():.2f} % y {tabla['atipicos_pct'].max():.2f} %: datos limpios por diseño, "
          "por eso los faltantes se simulan (10 %).")
    f = tabla.loc[tabla["razon_escalas_max_min"].idxmax()]
    print(f"  · Mayor dispersión de escalas: {f['dataset']} (razón {f['razon_escalas_max_min']:,.0f}); el escalado "
          "solo afecta a modelos sensibles a la escala (lineal, MLP), no a los árboles.")
    a = tabla.loc[tabla["asimetria_media_abs"].idxmax()]
    print(f"  · Mayor asimetría: {a['dataset']} ({a['asimetria_media_abs']:.2f}); anticipa atípicos que pueden "
          "desestabilizar a la red neuronal.")


# %% PASO 4 · Ejecución de punta a punta
# ------------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
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
    figura(escalas, minoritarias, SALIDA / "fig_eda_datasets.png")
    interpretar(tabla)
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
