# ==============================================================================
# audit_llm_generated.py · Auditoría estática de los pipelines escritos por LLM (Conjunto A)
# ==============================================================================
"""
Audita SOLO de forma estática (sin ejecutar) los scripts de 02_pipelines/llm_generated/<modelo>/.

    python 04_experiments/audit_llm_generated.py [carpeta]

Salida: results/llm_generated_audit.csv y una tabla por modelo con el % de scripts con fuga.
Estado en Capstone I: la carpeta aún no contiene scripts (se generan en Capstone II).
"""

# %% PASO 0 · Configuración del entorno e importaciones
# ------------------------------------------------------------------------------
# SEGURIDAD: el código escrito por un LLM es NO CONFIABLE; por eso aquí solo se LEE con el AST.
#   Ejecutarlo (verificador en ejecución) exigirá un contenedor aislado, sin red ni credenciales.
# ------------------------------------------------------------------------------
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "03_auditor"))
from ast_leakage_detector import analizar_archivo  # noqa: E402


# %% PASO 1 · Auditar cada script generado
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: 02_pipelines/llm_generated/<modelo>/<DATASET>_<FAMILIA>.py, escritos por
#   generate_with_llm.py con el mismo prompt estándar para todos los modelos.
# MÉTRICA: por modelo, n scripts, cuántos con fuga y su porcentaje (prevalencia de fuga).
# ------------------------------------------------------------------------------
def auditar_carpeta(carpeta):
    filas = []
    for ruta in sorted(carpeta.glob("*/*.py")):
        r = analizar_archivo(ruta)
        filas.append({"modelo_llm": ruta.parent.name, "script": ruta.name, "estado": r["estado"],
                      "tipos": "|".join(r["tipos"]), "n_hallazgos": len(r["hallazgos"])})
    return pd.DataFrame(filas)


# %% PASO 2 · Ejecución e interpretación
# ------------------------------------------------------------------------------
# CÓMO SE INTERPRETA: el % con fuga es una cota INFERIOR de la prevalencia real (el AST no ve alias
#   ni código dinámico) y depende del prompt; debe acompañarse de una muestra revisada a mano.
# ------------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    carpeta = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "02_pipelines" / "llm_generated"
    df = auditar_carpeta(carpeta)
    if df.empty:
        print(f"No hay scripts en {carpeta}/<modelo>/*.py. Genérelos con generate_with_llm.py (Capstone II).")
        return
    df.to_csv(RAIZ / "04_experiments" / "results" / "llm_generated_audit.csv", index=False)
    resumen = df.groupby("modelo_llm")["estado"].agg(
        n="count", con_fuga=lambda s: (s == "FUGA_DETECTADA").sum())
    resumen["pct_con_fuga"] = (100 * resumen["con_fuga"] / resumen["n"]).round(1)
    print(resumen.to_string())
    print("\nINTERPRETACIÓN")
    print("  · El porcentaje es una cota inferior de la prevalencia de fuga (análisis estático); "
          "valídelo con una muestra revisada a mano antes de comparar modelos.")


if __name__ == "__main__":
    main()
