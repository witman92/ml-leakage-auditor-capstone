# ==============================================================================
# statistical_analysis.py · FASE 4 (continuación): ¿qué familias de modelos sufren más la fuga?
# ==============================================================================
"""
Análisis estadístico del Sesgo de Optimismo ΔOpt por familia de modelos y tipo de fuga.

    python 04_experiments/statistical_analysis.py

Entrada : 04_experiments/results/delta_optimism_pairs.csv   (lo genera run_benchmark.py)
Salidas : delta_opt_by_family_leak.csv   media de ΔOpt por tarea × familia × fuga
          tests_by_family.csv            ¿ΔOpt > 0? Wilcoxon, t pareada, Holm e IC95 % bootstrap
          friedman_families.csv          ¿difieren las familias entre sí? (Friedman + Bonferroni)
          degenerate_pairs.csv           pares excluidos por divergencia numérica
          fig_delta_opt_heatmap.png      mapa de calor familia × fuga
          fig_delta_opt_bars.png         ΔOpt medio por familia y fuga, con dispersión
"""

# %% PASO 0 · Configuración del entorno e importaciones
# ------------------------------------------------------------------------------
# QUÉ HACE   : importa librerías y fija constantes del análisis (métrica principal por tarea,
#              orden de familias y fugas, semilla del bootstrap).
# MÉTRICA PRINCIPAL: clasificación → ROC-AUC; regresión → R². ΔOpt está en las unidades de la
#              métrica: "+0,05" = la fuga infló el AUC (o el R²) en 0,05 puntos.
# ------------------------------------------------------------------------------
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RESULTADOS = Path(__file__).resolve().parent / "results"
SEMILLA = 42
N_BOOTSTRAP = 10_000
METRICA_PRINCIPAL = {"clasificacion": "roc_auc", "regresion": "r2"}
ORDEN_FAMILIAS = ["RF", "GB", "LIN", "MLP"]
NOMBRE_FAMILIA = {"RF": "RandomForest", "GB": "HistGradBoosting",
                  "LIN": "Lineal (LogReg/Ridge)", "MLP": "Red neuronal MLP"}
FUGAS_SIMPLES = ["ESCALADO", "IMPUTACION", "SELECCION"]
ORDEN_FUGAS = FUGAS_SIMPLES + ["COMBINADA"]


# %% PASO 1 · Carga de los pares y exclusión de degenerados
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: delta_optimism_pairs.csv (una fila por dataset × familia × fuga × métrica).
# LÓGICA:
#   - Se conserva solo la métrica principal de cada tarea.
#   - Se excluyen los PARES DEGENERADOS (algún pliegue con R² < −1: el modelo divergió). Miden
#     inestabilidad numérica, no fuga; se listan aparte para que la exclusión sea auditable.
# ------------------------------------------------------------------------------
def cargar_pares(ruta=RESULTADOS / "delta_optimism_pairs.csv"):
    """Lee los pares y se queda con la métrica principal de cada tarea."""
    df = pd.read_csv(ruta)
    return df[df.apply(lambda f: f["metrica"] == METRICA_PRINCIPAL[f["tarea"]], axis=1)].copy()


def sin_degenerados(df):
    """Quita los pares donde un modelo divergió (ver run_benchmark.UMBRAL_DEGENERADO)."""
    return df[~df["degenerado"]].copy()


# %% PASO 2 · Métrica descriptiva: ΔOpt medio por familia y fuga
# ------------------------------------------------------------------------------
# LÓGICA: promedio, desviación y n de ΔOpt sobre los datasets de cada tarea (3 por tarea, o 2 si
#         se excluyó un par degenerado). Alimenta la Tabla de ΔOpt y el mapa de calor.
# ------------------------------------------------------------------------------
def tabla_medias(df):
    """ΔOpt medio por tarea × familia × fuga (promedio sobre los datasets)."""
    t = (df.groupby(["tarea", "familia", "fuga"])["delta_opt"].agg(["mean", "std", "count"]).reset_index()
           .rename(columns={"mean": "delta_opt_medio", "std": "desv", "count": "n_datasets"}))
    return t.round(5)


# %% PASO 3 · Inferencia por familia: ¿ΔOpt > 0?
# ------------------------------------------------------------------------------
# DISEÑO:
#   - Unidad de análisis = PAR (dataset × fuga simple). Los 5 pliegues de un par NO son
#     independientes, por eso no se usan como muestras separadas. n = 9 pares por familia y tarea.
#   - Solo fugas simples: COMBINADA repite los mismos mecanismos y contaría dos veces lo mismo.
#   - Wilcoxon de rangos con signo, unilateral (H1: ΔOpt > 0): no supone normalidad (Demšar, 2006).
#     Los pares con ΔOpt exactamente 0 se descartan (esperable en árboles con escalado).
#   - t pareada unilateral como contraste paramétrico; d de Cohen (dz) como tamaño del efecto.
#   - MULTIPLICIDAD: se hacen 8 pruebas de Wilcoxon (2 tareas × 4 familias). Se reporta el p
#     ajustado por Holm (step-down): ordena los p de menor a mayor, multiplica el i-ésimo por
#     (m − i + 1), impone monotonía y recorta a 1.
#   - IC95 % bootstrap percentil de la media de ΔOpt (10 000 remuestreos, semilla 42): rango
#     plausible del efecto medio, que con n = 9 es más informativo que el valor p.
# ------------------------------------------------------------------------------
def holm(p_valores):
    """Ajuste de Holm (step-down) de un vector de valores p; NaN se conserva."""
    p = np.asarray(p_valores, float)
    ajustados = np.full_like(p, np.nan)
    validos = np.where(~np.isnan(p))[0]
    m = len(validos)
    orden = validos[np.argsort(p[validos], kind="mergesort")]
    acumulado = 0.0
    for i, idx in enumerate(orden):
        acumulado = max(acumulado, min(1.0, (m - i) * p[idx]))   # monotonía no decreciente
        ajustados[idx] = acumulado
    return ajustados


def ic_bootstrap(d, rng, n=N_BOOTSTRAP):
    """IC95 % percentil de la media de d por remuestreo con reemplazo."""
    medias = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return np.percentile(medias, [2.5, 97.5])


def tests_por_familia(df):
    """Wilcoxon unilateral, t pareada, Holm e IC95 % bootstrap por tarea y familia (fugas simples)."""
    filas = []
    rng = np.random.default_rng(SEMILLA)
    simples = df[df["fuga"].isin(FUGAS_SIMPLES)]
    for (tarea, familia), g in simples.groupby(["tarea", "familia"]):
        d = g["delta_opt"].to_numpy()
        no_cero = d[d != 0]
        try:
            p_w = stats.wilcoxon(no_cero, alternative="greater").pvalue if len(no_cero) >= 1 else np.nan
        except ValueError:
            p_w = np.nan
        t = stats.ttest_1samp(d, 0.0, alternative="greater")
        inf, sup = ic_bootstrap(d, rng)
        filas.append({"tarea": tarea, "familia": familia, "n_pares": len(d), "n_pares_no_cero": len(no_cero),
                      "delta_opt_medio": round(d.mean(), 5), "mediana": round(float(np.median(d)), 5),
                      "d_cohen_dz": round(d.mean() / d.std(ddof=1), 2) if d.std(ddof=1) > 0 else np.nan,
                      "p_wilcoxon_unilateral": round(p_w, 4), "p_t_unilateral": round(t.pvalue, 4),
                      "p_wilcoxon_crudo": p_w, "ic95_inf": round(float(inf), 5), "ic95_sup": round(float(sup), 5)})
    tabla = pd.DataFrame(filas)
    tabla["p_wilcoxon_holm"] = np.round(holm(tabla.pop("p_wilcoxon_crudo")), 4)
    return tabla


# %% PASO 4 · Comparación entre familias: prueba de Friedman
# ------------------------------------------------------------------------------
# LÓGICA: bloques = (dataset, fuga simple), tratamientos = las 4 familias (Demšar, 2006). Se
#   exigen bloques completos. Rango medio 1 = la familia que más optimismo sufre. Con 2 pruebas
#   (una por tarea) se reporta el p con corrección de Bonferroni: min(1, 2p).
# ------------------------------------------------------------------------------
def friedman_familias(df):
    """¿Difieren las 4 familias? Bloques = (dataset, fuga simple)."""
    filas = []
    simples = df[df["fuga"].isin(FUGAS_SIMPLES)]
    tareas = list(simples.groupby("tarea"))
    for tarea, g in tareas:
        ancho = (g.pivot_table(index=["dataset", "fuga"], columns="familia", values="delta_opt")[ORDEN_FAMILIAS]
                 .dropna())
        estadistico, p = stats.friedmanchisquare(*[ancho[f] for f in ORDEN_FAMILIAS])
        rangos = ancho.rank(axis=1, ascending=False).mean()
        filas.append({"tarea": tarea, "bloques": len(ancho), "chi2": round(estadistico, 3), "p": round(p, 4),
                      **{f"rango_medio_{f}": round(rangos[f], 2) for f in ORDEN_FAMILIAS},
                      "p_bonferroni": round(min(1.0, p * len(tareas)), 4)})
    return pd.DataFrame(filas)


# %% PASO 5 · Gráfico 1: mapa de calor familia × fuga
# ------------------------------------------------------------------------------
# QUÉ DIBUJA : dos paneles (clasificación, regresión); cada celda es el ΔOpt medio con su valor
#              escrito. Escala de rojos desde 0: más intenso = más optimismo.
# CÓMO LEERLO: filas = familias, columnas = tipos de fuga. Una fila clara en ESCALADO para los
#              árboles confirma que el escalado no altera su estructura.
# ------------------------------------------------------------------------------
def figura_heatmap(medias, ruta):
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


# %% PASO 6 · Gráfico 2: barras de ΔOpt con dispersión entre datasets
# ------------------------------------------------------------------------------
# QUÉ DIBUJA : ΔOpt medio por familia y fuga; la barra de error es la desviación entre datasets.
# CÓMO LEERLO: barras que cruzan el 0 con error grande = efecto no distinguible del ruido.
# ------------------------------------------------------------------------------
def figura_barras(df, ruta):
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


# %% PASO 7 · Interpretación analítica automática
# ------------------------------------------------------------------------------
# REGLAS DE LECTURA:
#   - Un efecto es "confirmatorio" solo si su p ajustado (Holm) < 0,05; si solo el p crudo lo es,
#     se reporta como EXPLORATORIO.
#   - Un IC95 % que incluye el 0 indica que, con 9 pares, no se puede descartar ΔOpt medio nulo.
# ------------------------------------------------------------------------------
def interpretar(tests, friedman):
    print("\nINTERPRETACIÓN")
    for _, f in tests.iterrows():
        if f["p_wilcoxon_holm"] < 0.05:
            veredicto = "significativo tras Holm (confirmatorio)"
        elif f["p_wilcoxon_unilateral"] < 0.05:
            veredicto = "significativo sin corregir, NO tras Holm (exploratorio)"
        else:
            veredicto = "no significativo"
        incluye_cero = f["ic95_inf"] <= 0 <= f["ic95_sup"]
        print(f"  · {f['tarea']:13s} {f['familia']:3s}: ΔOpt medio {f['delta_opt_medio']:+.4f}, "
              f"IC95 % [{f['ic95_inf']:+.4f}; {f['ic95_sup']:+.4f}]{' (incluye 0)' if incluye_cero else ''}; "
              f"{veredicto}.")
    for _, f in friedman.iterrows():
        print(f"  · Friedman {f['tarea']}: p = {f['p']:.3f}; Bonferroni = {f['p_bonferroni']:.3f} → "
              f"{'las familias difieren' if f['p_bonferroni'] < 0.05 else 'sin evidencia de diferencias tras corregir'}.")


# %% PASO 8 · Ejecución de punta a punta
# ------------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    todos = cargar_pares()
    todos[todos["degenerado"]].to_csv(RESULTADOS / "degenerate_pairs.csv", index=False)
    df = sin_degenerados(todos)
    print(f"Pares con la métrica principal: {len(todos)} · degenerados excluidos: {int(todos['degenerado'].sum())}")
    medias, tests, friedman = tabla_medias(df), tests_por_familia(df), friedman_familias(df)
    medias.to_csv(RESULTADOS / "delta_opt_by_family_leak.csv", index=False)
    tests.to_csv(RESULTADOS / "tests_by_family.csv", index=False)
    friedman.to_csv(RESULTADOS / "friedman_families.csv", index=False)
    figura_heatmap(medias, RESULTADOS / "fig_delta_opt_heatmap.png")
    figura_barras(df, RESULTADOS / "fig_delta_opt_bars.png")
    print("=== ΔOpt medio (métrica principal) ===")
    print(medias.pivot_table(index=["tarea", "familia"], columns="fuga", values="delta_opt_medio")
                .reindex(columns=ORDEN_FUGAS).round(4).to_string())
    print("\n=== Tests por familia (fugas simples) ===")
    print(tests.to_string(index=False))
    print("\n=== Friedman entre familias ===")
    print(friedman.to_string(index=False))
    interpretar(tests, friedman)


if __name__ == "__main__":
    main()
