# ==============================================================================
# leakage_demo.py · Demostración mínima y visual de la fuga de datos (observación R3)
# ==============================================================================
"""
Demostración paso a paso del Sesgo de Optimismo  ΔOpt = M_leaky − M_clean.

El MISMO modelo, los MISMOS datos y los MISMOS pliegues se evalúan de dos maneras:
  A) CON FUGA : imputador, escalador y selector se ajustan con TODAS las filas antes de la
                validación cruzada (patrón típico del código generado por un LLM).
  B) SIN FUGA : los tres pasos van dentro de un sklearn.pipeline.Pipeline; en cada pliegue,
                .fit() solo ve el 80 % de entrenamiento.

Ejecución (VS Code: botón "Run" o celda a celda con "Run Cell"; Colab: pegar celda a celda):
    python 04_experiments/leakage_demo.py

Salidas:
    04_experiments/results/leakage_demo.csv       tabla con AUC, desviaciones y ΔOpt
    04_experiments/results/fig_leakage_demo.png   Figura 1.2 del informe
"""

# %% PASO 0 · Configuración del entorno e importaciones
# ------------------------------------------------------------------------------
# QUÉ HACE   : importa las librerías y fija la semilla única del proyecto (42).
# POR QUÉ    : con la semilla fija, dos ejecuciones producen exactamente las mismas cifras
#              (requisito de reproducibilidad: el autor debe poder defender cada número).
# DEPENDENCIAS: scikit-learn, numpy, pandas y matplotlib (versiones en requirements.txt).
# ------------------------------------------------------------------------------
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # dibuja directamente a archivo PNG (sin ventana):
                                            # funciona igual en VS Code, Colab y en un servidor CI
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

warnings.filterwarnings("ignore")           # avisos de convergencia del MLP: no cambian la lección

SEMILLA = 42                                # semilla única de todo el proyecto
K_VARIABLES = 8                             # SelectKBest conserva las 8 variables "mejores"
N_COLUMNAS_RUIDO = 500                      # columnas de ruido del escenario de señal nula
SALIDA = Path(__file__).resolve().parent / "results"


# %% PASO 1 · Definición del modelo (idéntico en ambos pipelines)
# ------------------------------------------------------------------------------
# QUÉ HACE   : construye una red neuronal MLP con hiperparámetros FIJOS.
# POR QUÉ    : para aislar el efecto causal de la fuga, lo único que puede variar entre A y B es
#              DÓNDE se ajusta el preprocesamiento. Mismo modelo + misma semilla = misma red.
#              Se elige una MLP porque es una familia sensible a la escala de las variables.
# ------------------------------------------------------------------------------
def modelo():
    """Devuelve una MLP nueva (sin entrenar) con la configuración fija del experimento."""
    return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEMILLA)


# %% PASO 2 · Función central: evaluar el mismo pipeline CON y SIN fuga
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: recibe una matriz X (con celdas NaN) y un vector y ya cargados (PASO 3).
# LÓGICA:
#   - Pliegues: StratifiedKFold(5, semilla 42). El MISMO objeto de pliegues se usa en A y B,
#     así ambos pipelines se evalúan exactamente sobre las mismas filas de prueba.
#   - A) CON FUGA: los tres .fit_transform() se ejecutan sobre las 569 filas ANTES de partir.
#       · KNNImputer   → rellena un NaN con el promedio de 5 vecinos, que pueden ser filas de prueba.
#       · StandardScaler → la media y la desviación incluyen las filas de prueba.
#       · SelectKBest  → elige variables mirando las etiquetas y de TODAS las filas (la más grave).
#     cross_val_score recibe una matriz YA contaminada: cada pliegue de prueba ya fue "visto".
#   - B) SIN FUGA: los mismos tres pasos dentro de un Pipeline. cross_val_score clona el Pipeline
#     en cada pliegue y llama .fit() solo con el 80 % de entrenamiento; la prueba solo se transforma.
# MÉTRICA: ROC-AUC por pliegue (5 valores por pipeline). ΔOpt = media(A) − media(B).
# ------------------------------------------------------------------------------
def comparar(X_faltante, y, k, etiqueta):
    """Devuelve (scores_leaky, scores_clean): 5 valores de ROC-AUC cada uno, mismos pliegues."""
    pliegues = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    # --- A) CON FUGA: los tres .fit() ocurren antes de dividir en pliegues ----------------
    X_imp = KNNImputer(n_neighbors=5).fit_transform(X_faltante)        # fuga por imputación
    X_esc = StandardScaler().fit_transform(X_imp)                      # fuga por escalado
    X_sel = SelectKBest(f_classif, k=k).fit_transform(X_esc, y)        # fuga por selección (usa y)
    leaky = cross_val_score(modelo(), X_sel, y, cv=pliegues, scoring="roc_auc")

    # --- B) SIN FUGA: todo dentro del Pipeline; se reajusta en cada pliegue ---------------
    limpio = Pipeline([("imputador", KNNImputer(n_neighbors=5)),
                       ("escalador", StandardScaler()),
                       ("selector", SelectKBest(f_classif, k=k)),
                       ("modelo", modelo())])
    clean = cross_val_score(limpio, X_faltante, y, cv=pliegues, scoring="roc_auc")

    print(f"[{etiqueta}] con fuga {leaky.mean():.4f} | sin fuga {clean.mean():.4f} | "
          f"ΔOpt = {leaky.mean() - clean.mean():+.4f}")
    return leaky, clean


# %% PASO 3 · Carga de datos y construcción de los dos escenarios
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: sklearn.datasets.load_breast_cancer (nativo de scikit-learn, sin internet):
#   569 tumores × 30 variables morfológicas; y = 0 (maligno) / 1 (benigno). Datos públicos y
#   anonimizados (Wisconsin Diagnostic Breast Cancer), usados solo con fines metodológicos.
# PREPARACIÓN:
#   - Se borra el 10 % de las celdas al azar (semilla 42) para que el imputador tenga celdas que rellenar.
#   - Escenario 1 "señal real": las etiquetas verdaderas (señal fuerte, AUC ≈ 0,99).
#   - Escenario 2 "señal nula" (CONTROL POSITIVO): se agregan 500 columnas de ruido gaussiano y
#     se PERMUTAN las etiquetas. Ninguna variable predice nada: la verdad es AUC = 0,5. Si el
#     pipeline con fuga da más de 0,5, ese exceso es optimismo puro creado por la fuga.
# NOTA DE REPRODUCIBILIDAD: el orden de las llamadas a `rng` es parte del diseño; no reordenar.
# ------------------------------------------------------------------------------
def construir_escenarios():
    """Devuelve {"señal real": (X, y), "señal nula": (X, y)} de forma determinista."""
    print("Cargando breast_cancer (nativo de scikit-learn)...")
    X, y = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(SEMILLA)

    X_faltante = X.copy()
    X_faltante[rng.random(X.shape) < 0.10] = np.nan        # 10 % de NaN, determinista

    y_azar = rng.permutation(y)                            # rompe la relación X → y
    X_ruido = np.hstack([X_faltante, rng.normal(size=(X.shape[0], N_COLUMNAS_RUIDO))])
    return {"señal real": (X_faltante, y), "señal nula": (X_ruido, y_azar)}


# %% PASO 4 · Métricas: tabla de resultados
# ------------------------------------------------------------------------------
# QUÉ CALCULA: por escenario, el ROC-AUC medio con y sin fuga, su desviación estándar entre
#              pliegues y ΔOpt = AUC_con_fuga − AUC_sin_fuga (positivo = la fuga infla la métrica).
# DÓNDE QUEDA: results/leakage_demo.csv (lo cita el informe, sección 1.2).
# ------------------------------------------------------------------------------
def tabla_resultados(resultados):
    filas = []
    for escenario, (leaky, clean) in resultados.items():
        nombre = escenario if escenario == "señal real" else "señal nula (verdad AUC=0,5)"
        filas.append({"escenario": nombre,
                      "auc_con_fuga": round(leaky.mean(), 4), "auc_sin_fuga": round(clean.mean(), 4),
                      "delta_opt": round(leaky.mean() - clean.mean(), 4),
                      "sd_con_fuga": round(leaky.std(), 4), "sd_sin_fuga": round(clean.std(), 4)})
    return pd.DataFrame(filas)


# %% PASO 5 · Gráfico explicativo (Figura 1.2 del informe)
# ------------------------------------------------------------------------------
# QUÉ DIBUJA : dos paneles de barras (a) señal real y (b) señal nula. Cada panel compara el AUC
#              medio con fuga (rojo) y sin fuga (azul); la barra de error es la desviación entre
#              los 5 pliegues y la línea punteada marca el azar (AUC = 0,5).
# DECISIONES : eje Y completo (0 a 1,05) para NO exagerar visualmente diferencias pequeñas;
#              el número exacto se escribe dentro de cada barra.
# CÓMO LEERLO: la distancia vertical entre la barra roja y la azul es ΔOpt.
# ------------------------------------------------------------------------------
def figura(resultados, ruta):
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    titulos = ("(a) Señal real", "(b) Señal nula (verdad: AUC = 0,5)")
    for eje, titulo, (leaky, clean) in zip(ejes, titulos, resultados.values()):
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
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


# %% PASO 6 · Interpretación analítica automática
# ------------------------------------------------------------------------------
# QUÉ HACE   : traduce las cifras a una conclusión en palabras, con reglas explícitas, para que la
#              lectura no dependa de la impresión visual del gráfico.
# CÓMO SE INTERPRETA:
#   - Señal real: ΔOpt suele ser muy pequeño. NO significa que la fuga sea inocua: con AUC ≈ 0,99
#     queda poco margen para inflar (efecto techo), y con variables muy informativas el selector
#     elige las mismas columnas con o sin fuga.
#   - Señal nula: cualquier AUC con fuga por encima de 0,5 es optimismo fabricado. Es la prueba de
#     que el mecanismo existe y de que ΔOpt lo detecta cuando la señal es débil frente al ruido.
# ------------------------------------------------------------------------------
def interpretar(tabla):
    real, nula = tabla.iloc[0], tabla.iloc[1]
    print("\nINTERPRETACIÓN")
    print(f"  · Señal real: ΔOpt = {real['delta_opt']:+.4f}. Con un AUC limpio de {real['auc_sin_fuga']:.3f} "
          "hay efecto techo: la fuga existe, pero casi no se ve.")
    print(f"  · Señal nula: el pipeline con fuga reporta AUC = {nula['auc_con_fuga']:.3f} cuando la verdad "
          f"es 0,5; el limpio reporta {nula['auc_sin_fuga']:.3f}. ΔOpt = {nula['delta_opt']:+.4f} es "
          "rendimiento ficticio creado solo por ajustar el preprocesamiento con todas las filas.")
    print("  · Conclusión: la gravedad de la fuga depende de la razón señal/ruido de los datos, "
          "no solo del tipo de error en el código.")


# %% PASO 7 · Ejecución de punta a punta
# ------------------------------------------------------------------------------
# Orquesta los pasos 3 → 2 → 4 → 5 → 6 y guarda los artefactos en results/.
# ------------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")            # tildes correctas en la consola de Windows
    escenarios = construir_escenarios()
    resultados = {nombre: comparar(X, y, k=K_VARIABLES, etiqueta=f"{nombre:16s}")
                  for nombre, (X, y) in escenarios.items()}

    tabla = tabla_resultados(resultados)
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / "leakage_demo.csv", index=False)
    figura(resultados, SALIDA / "fig_leakage_demo.png")

    print(tabla.to_string(index=False))
    interpretar(tabla)
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
