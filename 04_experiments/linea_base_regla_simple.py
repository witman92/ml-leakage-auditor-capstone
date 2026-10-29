"""
linea_base_regla_simple.py  ·  Línea base externa: ¿qué tanto aporta el auditor frente a una regla trivial?

POR QUÉ HACE FALTA  (rúbrica v2, criterio 4: «comparar contra una línea base»)
    Sin una referencia, "recall 0,89" no dice si es mucho o poco. Esta línea base es lo que haría
    un revisor apurado con `grep`: declara FUGA si el script llama .fit() o .fit_transform() de un
    transformador (Scaler, Imputer, SelectKBest, PCA...) y NO usa Pipeline / make_pipeline en
    ningún lugar. No entiende el orden del código, ni train_test_split, ni los alias.

    Es una regla escrita por el autor (no un linter de terceros): se declara así en el informe.

SALIDAS   results/linea_base_metricas.csv   results/linea_base_por_script.csv

    python 04_experiments/linea_base_regla_simple.py
"""
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

SALIDA = RAIZ / "04_experiments" / "results"
TRANSFORMADORES = r"(Scaler|Imputer|SelectKBest|SelectPercentile|PCA|VarianceThreshold|Normalizer|Encoder)"
AJUSTE = re.compile(r"\.fit(_transform)?\(")
CREA_TRANSFORMADOR = re.compile(TRANSFORMADORES)
USA_PIPELINE = re.compile(r"\b(make_pipeline|Pipeline)\b")


def solo_codigo(texto: str) -> str:
    """Quita comentarios, cadenas (docstrings) y líneas de import: deja solo el código que se ejecuta."""
    tokens = [t for t in tokenize.generate_tokens(io.StringIO(texto).readline)
              if t.type not in (tokenize.COMMENT, tokenize.STRING)]
    codigo = tokenize.untokenize(tokens)
    return "\n".join(l for l in codigo.splitlines() if not l.lstrip().startswith(("import ", "from ")))


def regla_simple(texto: str) -> bool:
    """True = «hay fuga». Ajusta un transformador a mano y el script no usa ningún Pipeline.

    Dos errores de la propia regla corregidos tras verlos (recall 0 en la primera corrida): contaba
    `import Pipeline` y la palabra «Pipeline» de un docstring como si fueran uso real."""
    codigo = solo_codigo(texto)
    return bool(AJUSTE.search(codigo) and CREA_TRANSFORMADOR.search(codigo) and not USA_PIPELINE.search(codigo))


def evaluar(conjunto, rutas, etiquetas):
    filas = [{"conjunto": conjunto, "script": r.name, "fuga_real": etiquetas[r.name],
              "regla_simple": regla_simple(r.read_text(encoding="utf-8"))} for r in rutas]
    return pd.DataFrame(filas)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Banco de control: la etiqueta está en el nombre del archivo (_leaky / _clean)
    banco = sorted((RAIZ / "02_pipelines" / "controlled_pairs").glob("*.py"))
    et_banco = {r.name: r.stem.endswith("_leaky") for r in banco}
    # Evaluación independiente: etiquetas manuales en esperado.json
    casos = RAIZ / "08_pruebas" / "casos_manuales"
    esperado = json.loads((casos / "esperado.json").read_text(encoding="utf-8"))
    et_indep = {n: i["fuga"] for n, i in esperado.items()}

    por_script = pd.concat([evaluar("banco (192)", banco, et_banco),
                            evaluar("independiente (13)", [casos / n for n in esperado], et_indep)])
    por_script.to_csv(SALIDA / "linea_base_por_script.csv", index=False)

    metricas = pd.DataFrame([{"conjunto": c, "detector": "Regla simple (línea base)",
                              **metricas_deteccion(g["fuga_real"], g["regla_simple"])}
                             for c, g in por_script.groupby("conjunto", sort=False)])
    metricas.to_csv(SALIDA / "linea_base_metricas.csv", index=False)
    print(metricas.to_string(index=False))
    indep = por_script[por_script["conjunto"].str.startswith("indep")]
    print("\nErrores de la regla simple en la evaluación independiente:")
    print(indep[indep["fuga_real"] != indep["regla_simple"]][["script", "fuga_real", "regla_simple"]].to_string(index=False))


if __name__ == "__main__":
    main()
