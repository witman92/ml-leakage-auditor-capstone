# ==============================================================================
# independent_evaluation.py · ¿Generaliza el auditor a código que NO generó él mismo?
# ==============================================================================
"""
Evaluación independiente del auditor con 13 scripts escritos a mano.

    python 04_experiments/independent_evaluation.py

Salidas: results/independent_evaluation.csv (por script)
         results/independent_evaluation_metrics.csv (por detector, con IC95 % del recall)
"""

# %% PASO 0 · Configuración del entorno e importaciones
# ------------------------------------------------------------------------------
# POR QUÉ HACE FALTA: el banco de 192 scripts lo produce un generador escrito por el mismo autor
#   del auditor, así que su 100 % era esperable (evaluación circular). Aquí se usan estilos que
#   el generador no produce: make_pipeline, DataFrames, funciones auxiliares, alias de importación,
#   variables X_te, fit y transform por separado.
# ------------------------------------------------------------------------------
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import beta

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "03_auditor"))
sys.path.insert(0, str(RAIZ / "04_experiments"))
from ast_leakage_detector import analizar_archivo          # noqa: E402
from pipeline_fixer import corregir_archivo                # noqa: E402
from runtime_verifier import ejecutar_con_verificador      # noqa: E402
from run_benchmark import metricas_deteccion               # noqa: E402  (misma función de métricas)

CASOS = RAIZ / "08_tests" / "handwritten_cases"
SALIDA = RAIZ / "04_experiments" / "results"


# %% PASO 1 · Intervalo de confianza exacto para el recall
# ------------------------------------------------------------------------------
# LÓGICA: Clopper–Pearson al 95 % para una proporción binomial k/n (aquí k = TP, n = TP + FN).
#   Es exacto (no asintótico), adecuado para n pequeño como n = 9 fugas reales.
#   inf = Beta(α/2; k, n−k+1) ; sup = Beta(1−α/2; k+1, n−k)   (con k = 0 → inf = 0; k = n → sup = 1).
# ------------------------------------------------------------------------------
def clopper_pearson(k, n, alfa=0.05):
    """IC exacto de una proporción k/n. Devuelve (inf, sup) redondeados a 4 decimales."""
    if n == 0:
        return float("nan"), float("nan")
    inf = 0.0 if k == 0 else beta.ppf(alfa / 2, k, n - k + 1)
    sup = 1.0 if k == n else beta.ppf(1 - alfa / 2, k + 1, n - k)
    return round(float(inf), 4), round(float(sup), 4)


def metricas_con_ic(esperado, predicho):
    """metricas_deteccion + IC95 % Clopper–Pearson del recall."""
    m = metricas_deteccion(esperado, predicho)
    m["recall_ic95_inf"], m["recall_ic95_sup"] = clopper_pearson(m["TP"], m["TP"] + m["FN"])
    return m


# %% PASO 2 · Auditar cada caso escrito a mano
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: 08_tests/handwritten_cases/*.py y su etiqueta real en expected_labels.json
#   (la etiqueta la fijó el autor leyendo el código; es independiente del generador del banco).
# LÓGICA: para cada script se obtienen tres veredictos (AST, Runtime, Híbrido = AST o Runtime) y
#   se intenta la reparación automática.
# ------------------------------------------------------------------------------
def evaluar_casos():
    esperado = json.loads((CASOS / "expected_labels.json").read_text(encoding="utf-8"))
    filas = []
    for nombre, info in esperado.items():
        ruta = CASOS / nombre
        ast_fuga = analizar_archivo(ruta)["estado"] == "FUGA_DETECTADA"
        runtime_fuga = ejecutar_con_verificador(ruta)["fuga"]
        reparacion = corregir_archivo(ruta)
        filas.append({"script": nombre, "fuga_real": info["fuga"], "ast": ast_fuga, "runtime": runtime_fuga,
                      "hibrido": ast_fuga or runtime_fuga, "reparado_auto": reparacion["reparado"],
                      "descripcion": info["descripcion"]})
    return pd.DataFrame(filas)


# %% PASO 3 · Interpretación analítica automática
# ------------------------------------------------------------------------------
# CÓMO SE INTERPRETA: con n = 9 fugas el IC95 % del recall es ancho; un 9/9 NO permite afirmar un
#   umbral de 0,80 si el límite inferior queda por debajo. Que AST y Runtime fallen en casos
#   distintos respalda el diseño híbrido.
# ------------------------------------------------------------------------------
def interpretar(df, metricas):
    print("\nINTERPRETACIÓN")
    for _, m in metricas.iterrows():
        print(f"  · {m['detector']:8s}: recall {m['recall']:.2f}, IC95 % [{m['recall_ic95_inf']:.2f}; "
              f"{m['recall_ic95_sup']:.2f}], FP = {m['FP']}.")
    fallos_ast = set(df[df["fuga_real"] & ~df["ast"]]["script"])
    fallos_run = set(df[df["fuga_real"] & ~df["runtime"]]["script"])
    print(f"  · El AST falla en {sorted(fallos_ast)}; el Runtime en {sorted(fallos_run)}. "
          f"Fallos compartidos: {sorted(fallos_ast & fallos_run) or 'ninguno'} → la combinación es más robusta.")
    hib = metricas.iloc[-1]
    print(f"  · Umbral OE2 (recall ≥ 0,80): el límite inferior del IC es {hib['recall_ic95_inf']:.2f}; "
          f"{'se puede' if hib['recall_ic95_inf'] >= 0.80 else 'NO se puede'} afirmar con n = {hib['TP'] + hib['FN']}.")


# %% PASO 4 · Ejecución de punta a punta
# ------------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    df = evaluar_casos()
    SALIDA.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA / "independent_evaluation.csv", index=False)
    metricas = pd.DataFrame([{"detector": n, **metricas_con_ic(df["fuga_real"], df[c])}
                             for n, c in (("AST", "ast"), ("Runtime", "runtime"), ("Híbrido", "hibrido"))])
    metricas.to_csv(SALIDA / "independent_evaluation_metrics.csv", index=False)
    print(df[["script", "fuga_real", "ast", "runtime", "hibrido", "reparado_auto"]].to_string(index=False))
    print()
    print(metricas.to_string(index=False))
    leaky = df[df["fuga_real"]]
    print(f"\nReparación automática: {int(leaky['reparado_auto'].sum())}/{len(leaky)} de los scripts con fuga")
    interpretar(df, metricas)


if __name__ == "__main__":
    main()
