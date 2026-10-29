"""
evaluacion_independiente.py  ·  ¿Generaliza el auditor a código que NO generó él mismo?

POR QUÉ HACE FALTA
    El banco de 192 scripts (run_benchmark.py) lo produce un generador escrito por el mismo equipo que
    escribió el auditor: que acierte el 100 % era esperable (evaluación circular). Aquí se evalúa con
    13 scripts escritos A MANO (08_pruebas/casos_manuales), con estilos que el generador no produce
    (make_pipeline, DataFrames, funciones auxiliares, alias de importación, X_te, fit+transform
    separados...). Cada uno trae su etiqueta real en esperado.json.

SALIDAS   results/evaluacion_independiente.csv (por script)  y  results/evaluacion_independiente_metricas.csv

    python 04_experiments/evaluacion_independiente.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "03_auditor"))
sys.path.insert(0, str(RAIZ / "04_experiments"))
from ast_leakage_detector import analizar_archivo          # noqa: E402
from pipeline_fixer import corregir_archivo                # noqa: E402
from runtime_verifier import ejecutar_con_verificador      # noqa: E402
from run_benchmark import metricas_deteccion               # noqa: E402  (misma función de métricas)

CASOS = RAIZ / "08_pruebas" / "casos_manuales"
SALIDA = RAIZ / "04_experiments" / "results"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    esperado = json.loads((CASOS / "esperado.json").read_text(encoding="utf-8"))
    filas = []
    for nombre, info in esperado.items():
        ruta = CASOS / nombre
        ast_fuga = analizar_archivo(ruta)["estado"] == "FUGA_DETECTADA"
        runtime_fuga = ejecutar_con_verificador(ruta)["fuga"]
        reparacion = corregir_archivo(ruta)
        filas.append({"script": nombre, "fuga_real": info["fuga"], "ast": ast_fuga, "runtime": runtime_fuga,
                      "hibrido": ast_fuga or runtime_fuga, "reparado_auto": reparacion["reparado"],
                      "descripcion": info["descripcion"]})
    df = pd.DataFrame(filas)
    df.to_csv(SALIDA / "evaluacion_independiente.csv", index=False)
    metricas = pd.DataFrame([{"detector": n, **metricas_deteccion(df["fuga_real"], df[c])}
                             for n, c in (("AST", "ast"), ("Runtime", "runtime"), ("Híbrido", "hibrido"))])
    metricas.to_csv(SALIDA / "evaluacion_independiente_metricas.csv", index=False)
    print(df[["script", "fuga_real", "ast", "runtime", "hibrido", "reparado_auto"]].to_string(index=False))
    print()
    print(metricas.to_string(index=False))
    leaky = df[df["fuga_real"]]
    print(f"\nReparación automática: {int(leaky['reparado_auto'].sum())}/{len(leaky)} de los scripts con fuga")


if __name__ == "__main__":
    main()
