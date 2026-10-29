"""
auditar_llm_generados.py  ·  Audita (solo estático, sin ejecutar) los scripts de 02_pipelines/llm_generated/.

    python 04_experiments/auditar_llm_generados.py [carpeta]

Imprime una tabla por modelo con el porcentaje de scripts con fuga y los tipos encontrados, y guarda
results/llm_generados_auditoria.csv. NO ejecuta el código generado por el LLM (es código no confiable).
Mientras la carpeta esté vacía (estado actual de Capstone I) solo avisa que no hay scripts.
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "03_auditor"))
from ast_leakage_detector import analizar_archivo  # noqa: E402


def main():
    carpeta = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "02_pipelines" / "llm_generated"
    filas = []
    for ruta in sorted(carpeta.glob("*/*.py")):
        r = analizar_archivo(ruta)
        filas.append({"modelo_llm": ruta.parent.name, "script": ruta.name, "estado": r["estado"],
                      "tipos": "|".join(r["tipos"]), "n_hallazgos": len(r["hallazgos"])})
    if not filas:
        print(f"No hay scripts en {carpeta}/<modelo>/*.py. Genérelos con generar_con_llm.py (Capstone II).")
        return
    df = pd.DataFrame(filas)
    df.to_csv(RAIZ / "04_experiments" / "results" / "llm_generados_auditoria.csv", index=False)
    resumen = df.groupby("modelo_llm")["estado"].agg(
        n="count", con_fuga=lambda s: (s == "FUGA_DETECTADA").sum())
    resumen["pct_con_fuga"] = (100 * resumen["con_fuga"] / resumen["n"]).round(1)
    print(resumen.to_string())


if __name__ == "__main__":
    main()
