"""
statistical_analysis.py  ·  PASO 4 (continuación): ¿qué familias de modelos sufren más la fuga?

ENTRADA   04_experiments/results/delta_optimism_pairs.csv   (lo genera run_benchmark.py)
SALIDAS   delta_opt_por_familia_fuga.csv   media de ΔOpt por familia × tipo de fuga
          tests_por_familia.csv            ¿ΔOpt > 0 de forma significativa? (Wilcoxon y t pareada)
          friedman_familias.csv            ¿difieren las familias entre sí? (Friedman)
          fig_delta_opt_heatmap.png        mapa de calor familia × fuga
          fig_delta_opt_barras.png         ΔOpt medio por familia y fuga, con dispersión

MÉTRICA PRINCIPAL      clasificación -> ROC-AUC      regresión -> R²
(ΔOpt está en las unidades de la métrica: "+0.05" = la fuga infló el AUC en 0,05 puntos.)

DISEÑO ESTADÍSTICO (corto y honesto)
  * Unidad de análisis: cada PAR (dataset × tipo de fuga). Los 5 pliegues de un mismo par NO
    son independientes, por eso no se usan como muestras separadas.
  * Se usan solo las fugas "simples" (ESCALADO, IMPUTACION, SELECCION): COMBINADA repite esos
    mecanismos y contaría dos veces la misma información. Se reporta aparte en la tabla de medias.
  * Con 3 datasets × 3 fugas hay n = 9 pares por familia y tarea: es poco. Se interpreta el
    tamaño del efecto antes que el valor p, y no se generaliza más allá de estos datasets.
  * PARES DEGENERADOS (algún pliegue con R² < -1, el modelo divergió) se excluyen y se listan en
    pares_degenerados.csv: miden inestabilidad numérica, no fuga.
  * Wilcoxon de rangos con signo (unilateral, H1: ΔOpt > 0): no supone normalidad
    (Wilcoxon, 1945; Demšar, 2006). Los pares con ΔOpt exactamente 0 se descartan (esperable
    en árboles con escalado, ya que el escalado no cambia su estructura).
  * Friedman (Demšar, 2006): compara las 4 familias sobre los mismos 9 bloques.

    python 04_experiments/statistical_analysis.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RESULTADOS = Path(__file__).resolve().parent / "results"
METRICA_PRINCIPAL = {"clasificacion": "roc_auc", "regresion": "r2"}
ORDEN_FAMILIAS = ["RF", "GB", "LIN", "MLP"]
NOMBRE_FAMILIA = {"RF": "RandomForest", "GB": "HistGradBoosting",
                  "LIN": "Lineal (LogReg/Ridge)", "MLP": "Red neuronal MLP"}
FUGAS_SIMPLES = ["ESCALADO", "IMPUTACION", "SELECCION"]
ORDEN_FUGAS = FUGAS_SIMPLES + ["COMBINADA"]


def cargar_pares(ruta=RESULTADOS / "delta_optimism_pairs.csv"):
    """Lee los pares y se queda con la métrica principal de cada tarea."""
    df = pd.read_csv(ruta)
    return df[df.apply(lambda f: f["metrica"] == METRICA_PRINCIPAL[f["tarea"]], axis=1)].copy()


def sin_degenerados(df):
    """Quita los pares donde un modelo divergió (ver run_benchmark.UMBRAL_DEGENERADO)."""
    return df[~df["degenerado"]].copy()


def tabla_medias(df):
    """ΔOpt medio por tarea × familia × fuga (promedio sobre los datasets)."""
    t = (df.groupby(["tarea", "familia", "fuga"])["delta_opt"].agg(["mean", "std", "count"]).reset_index()
           .rename(columns={"mean": "delta_opt_medio", "std": "desv", "count": "n_datasets"}))
    return t.round(5)


def tests_por_familia(df):
    """Wilcoxon unilateral y t pareada (H1: ΔOpt > 0) por tarea y familia, solo fugas simples."""
    filas = []
    simples = df[df["fuga"].isin(FUGAS_SIMPLES)]
    for (tarea, familia), g in simples.groupby(["tarea", "familia"]):
        d = g["delta_opt"].to_numpy()
        no_cero = d[d != 0]
        try:
            p_w = stats.wilcoxon(no_cero, alternative="greater").pvalue if len(no_cero) >= 1 else np.nan
        except ValueError:
            p_w = np.nan
        t = stats.ttest_1samp(d, 0.0, alternative="greater")
        filas.append({"tarea": tarea, "familia": familia, "n_pares": len(d), "n_pares_no_cero": len(no_cero),
                      "delta_opt_medio": round(d.mean(), 5), "mediana": round(float(np.median(d)), 5),
                      "d_cohen_dz": round(d.mean() / d.std(ddof=1), 2) if d.std(ddof=1) > 0 else np.nan,
                      "p_wilcoxon_unilateral": round(p_w, 4), "p_t_unilateral": round(t.pvalue, 4)})
    return pd.DataFrame(filas)


def friedman_familias(df):
    """¿Difieren las 4 familias? Bloques = (dataset, fuga simple)."""
    filas = []
    simples = df[df["fuga"].isin(FUGAS_SIMPLES)]
    for tarea, g in simples.groupby("tarea"):
        ancho = (g.pivot_table(index=["dataset", "fuga"], columns="familia", values="delta_opt")[ORDEN_FAMILIAS]
                 .dropna())                       # Friedman exige bloques completos (las 4 familias)
        estadistico, p = stats.friedmanchisquare(*[ancho[f] for f in ORDEN_FAMILIAS])
        rangos = ancho.rank(axis=1, ascending=False).mean()     # rango 1 = la que más optimismo sufre
        filas.append({"tarea": tarea, "bloques": len(ancho), "chi2": round(estadistico, 3), "p": round(p, 4),
                      **{f"rango_medio_{f}": round(rangos[f], 2) for f in ORDEN_FAMILIAS}})
    return pd.DataFrame(filas)


def figura_heatmap(medias, ruta):
    """Figura: mapa de calor familia × fuga, una mitad por tarea."""
    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.6))
    for eje, tarea in zip(ejes, ("clasificacion", "regresion")):
        m = (medias[medias["tarea"] == tarea].pivot(index="familia", columns="fuga", values="delta_opt_medio")
             .reindex(index=ORDEN_FAMILIAS, columns=ORDEN_FUGAS))
        lim = float(np.nanmax(np.abs(m.values))) or 1.0
        imagen = eje.imshow(m.values, cmap="Reds", vmin=0, vmax=lim)
        eje.set_xticks(range(len(ORDEN_FUGAS)), ORDEN_FUGAS, rotation=20)
        eje.set_yticks(range(len(ORDEN_FAMILIAS)), [NOMBRE_FAMILIA[f] for f in ORDEN_FAMILIAS])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                eje.text(j, i, f"{m.values[i, j]:+.3f}", ha="center", va="center", fontsize=9,
                         color="white" if m.values[i, j] > 0.6 * lim else "black")
        eje.set_title(f"ΔOpt en {METRICA_PRINCIPAL[tarea].upper()} ({tarea})")
        fig.colorbar(imagen, ax=eje, shrink=0.8)
    plt.tight_layout()
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def figura_barras(df, ruta):
    """Figura: ΔOpt medio por familia y fuga, con la desviación entre datasets."""
    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
    ancho = 0.2
    for eje, tarea in zip(ejes, ("clasificacion", "regresion")):
        sub = df[df["tarea"] == tarea]
        for i, fam in enumerate(ORDEN_FAMILIAS):
            medias, desv = [], []
            for fuga in ORDEN_FUGAS:
                d = sub[(sub["familia"] == fam) & (sub["fuga"] == fuga)]["delta_opt"]
                medias.append(d.mean())
                desv.append(d.std())
            eje.bar(np.arange(len(ORDEN_FUGAS)) + (i - 1.5) * ancho, medias, ancho, yerr=desv, capsize=2,
                    label=NOMBRE_FAMILIA[fam])
        eje.axhline(0, color="black", lw=0.8)
        eje.set_xticks(range(len(ORDEN_FUGAS)), ORDEN_FUGAS)
        eje.set_ylabel(f"ΔOpt ({METRICA_PRINCIPAL[tarea].upper()})")
        eje.set_title(f"{tarea.capitalize()} (media ± desv. entre datasets)")
    ejes[0].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    todos = cargar_pares()
    todos[todos["degenerado"]].to_csv(RESULTADOS / "pares_degenerados.csv", index=False)
    df = sin_degenerados(todos)
    print(f"Pares con la métrica principal: {len(todos)} · degenerados excluidos: {int(todos['degenerado'].sum())}")
    medias, tests, friedman = tabla_medias(df), tests_por_familia(df), friedman_familias(df)
    medias.to_csv(RESULTADOS / "delta_opt_por_familia_fuga.csv", index=False)
    tests.to_csv(RESULTADOS / "tests_por_familia.csv", index=False)
    friedman.to_csv(RESULTADOS / "friedman_familias.csv", index=False)
    figura_heatmap(medias, RESULTADOS / "fig_delta_opt_heatmap.png")
    figura_barras(df, RESULTADOS / "fig_delta_opt_barras.png")
    print("=== ΔOpt medio (métrica principal) ===")
    print(medias.pivot_table(index=["tarea", "familia"], columns="fuga", values="delta_opt_medio")
                .reindex(columns=ORDEN_FUGAS).round(4).to_string())
    print("\n=== Tests por familia (fugas simples) ===")
    print(tests.to_string(index=False))
    print("\n=== Friedman entre familias ===")
    print(friedman.to_string(index=False))


if __name__ == "__main__":
    main()
