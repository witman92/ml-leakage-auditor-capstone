# ==============================================================================
# baseline_simple_rule.py · Línea base externa: ¿cuánto aporta el auditor frente a una regla trivial?
# ==============================================================================
"""
Línea base del detector: una regla tipo `grep` que cualquier revisor podría aplicar.

    python 04_experiments/baseline_simple_rule.py

Salidas: results/baseline_metrics.csv   y   results/baseline_per_script.csv
"""

# %% PASO 0 · Configuración del entorno e importaciones
# ------------------------------------------------------------------------------
# POR QUÉ HACE FALTA (rúbrica, criterio 4): sin una referencia, "recall 0,89" no dice si es mucho
#   o poco. La regla declara FUGA si el script llama .fit()/.fit_transform() de un transformador y
#   NO usa Pipeline / make_pipeline en ningún lugar. No entiende el orden del código, ni
#   train_test_split, ni los alias. La escribió el autor (no es un linter de terceros).
# ------------------------------------------------------------------------------
import io
import json
import re
import sys
import tokenize
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "04_experiments"))
from run_benchmark import metricas_deteccion               # noqa: E402  (misma función de métricas)
from independent_evaluation import clopper_pearson         # noqa: E402  (mismo IC exacto)

SALIDA = RAIZ / "04_experiments" / "results"
TRANSFORMADORES = r"(Scaler|Imputer|SelectKBest|SelectPercentile|PCA|VarianceThreshold|Normalizer|Encoder)"
AJUSTE = re.compile(r"\.fit(_transform)?\(")
CREA_TRANSFORMADOR = re.compile(TRANSFORMADORES)
USA_PIPELINE = re.compile(r"\b(make_pipeline|Pipeline)\b")


# %% PASO 1 · Limpieza del texto: dejar solo código ejecutable
# ------------------------------------------------------------------------------
# LÓGICA: con el tokenizador de Python se eliminan comentarios y cadenas (docstrings), y luego las
#   líneas de import. Así la palabra "Pipeline" dentro de un comentario o de un import no cuenta.
# ------------------------------------------------------------------------------
def solo_codigo(texto: str) -> str:
    tokens = [t for t in tokenize.generate_tokens(io.StringIO(texto).readline)
              if t.type not in (tokenize.COMMENT, tokenize.STRING)]
    codigo = tokenize.untokenize(tokens)
    return "\n".join(l for l in codigo.splitlines() if not l.lstrip().startswith(("import ", "from ")))


# %% PASO 2 · La regla simple
# ------------------------------------------------------------------------------
# REGLA: fuga ⇔ (hay .fit/.fit_transform) ∧ (se nombra un transformador) ∧ (no se usa Pipeline).
# NOTA DE HONESTIDAD: la primera versión daba recall 0 porque contaba `import Pipeline` y la
#   palabra «Pipeline» de un docstring como uso real; se corrigió la REGLA (no el auditor).
# ------------------------------------------------------------------------------
def regla_simple(texto: str) -> bool:
    """True = «hay fuga»."""
    codigo = solo_codigo(texto)
    return bool(AJUSTE.search(codigo) and CREA_TRANSFORMADOR.search(codigo) and not USA_PIPELINE.search(codigo))


def evaluar(conjunto, rutas, etiquetas):
    filas = [{"conjunto": conjunto, "script": r.name, "fuga_real": etiquetas[r.name],
              "regla_simple": regla_simple(r.read_text(encoding="utf-8"))} for r in rutas]
    return pd.DataFrame(filas)


# %% PASO 3 · Interpretación analítica automática
# ------------------------------------------------------------------------------
# CÓMO SE INTERPRETA: si la regla trivial pierde fugas en el banco o marca falsos positivos en los
#   casos manuales, ese margen es lo que aporta entender el ORDEN y el FLUJO de datos del código.
#   Con n = 13 la ventaja es una señal, no una mejora demostrada.
# ------------------------------------------------------------------------------
def interpretar(metricas, indep):
    print("\nINTERPRETACIÓN")
    for _, m in metricas.iterrows():
        print(f"  · {m['conjunto']}: recall {m['recall']:.2f} (IC95 % [{m['recall_ic95_inf']:.2f}; "
              f"{m['recall_ic95_sup']:.2f}]), FP = {m['FP']}, FN = {m['FN']}.")
    errores = indep[indep["fuga_real"] != indep["regla_simple"]]
    print(f"  · Errores de la regla en los casos manuales: {', '.join(errores['script']) or 'ninguno'}. "
          "No distingue escalar ANTES o DESPUÉS de partir, que es justo lo que define la fuga.")


# %% PASO 4 · Ejecución de punta a punta
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS:
#   - Banco de control (192): la etiqueta está en el nombre del archivo (_leaky / _clean).
#   - Evaluación independiente (13): etiquetas en 08_tests/handwritten_cases/expected_labels.json.
# ------------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    banco = sorted((RAIZ / "02_pipelines" / "controlled_pairs").glob("*.py"))
    et_banco = {r.name: r.stem.endswith("_leaky") for r in banco}
    casos = RAIZ / "08_tests" / "handwritten_cases"
    esperado = json.loads((casos / "expected_labels.json").read_text(encoding="utf-8"))
    et_indep = {n: i["fuga"] for n, i in esperado.items()}

    por_script = pd.concat([evaluar("banco (192)", banco, et_banco),
                            evaluar("independiente (13)", [casos / n for n in esperado], et_indep)])
    SALIDA.mkdir(parents=True, exist_ok=True)
    por_script.to_csv(SALIDA / "baseline_per_script.csv", index=False)

    filas = []
    for c, g in por_script.groupby("conjunto", sort=False):
        m = metricas_deteccion(g["fuga_real"], g["regla_simple"])
        m["recall_ic95_inf"], m["recall_ic95_sup"] = clopper_pearson(m["TP"], m["TP"] + m["FN"])
        filas.append({"conjunto": c, "detector": "Regla simple (línea base)", **m})
    metricas = pd.DataFrame(filas)
    metricas.to_csv(SALIDA / "baseline_metrics.csv", index=False)
    print(metricas.to_string(index=False))
    indep = por_script[por_script["conjunto"].str.startswith("indep")]
    interpretar(metricas, indep)


if __name__ == "__main__":
    main()
