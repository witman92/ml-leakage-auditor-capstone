# ==============================================================================
# run_all.py · Reproduce TODO el experimento con un solo comando
# ==============================================================================
"""
    python run_all.py            # completo (≈ 5-8 min)
    python run_all.py --rapido   # 2 datasets × 2 familias (≈ 30 s), para probar que todo funciona

Protocolo de 5 fases (cada paso es un script independiente que también se puede leer y ejecutar solo):

  FASE 0  01_data/sklearn_loader.py                       carga de los 6 datasets (prueba rápida)
  FASE 1  02_pipelines/generate_controlled_pairs.py       escribe los 96 pares leaky/clean (192 scripts)
  FASE 2-4 04_experiments/run_benchmark.py                audita, verifica, calcula ΔOpt y repara
  EDA     04_experiments/eda_datasets.py                  análisis exploratorio de los datasets
  ESTAD.  04_experiments/statistical_analysis.py          pruebas por familia (Holm, IC95 %), Friedman
  SENSIB. 04_experiments/noise_sensitivity.py             control positivo: selección vs. ruido
  DEMO    04_experiments/leakage_demo.py                  ejemplo visual de fuga (R3)
  BASE    04_experiments/baseline_simple_rule.py          línea base: regla simple
  INDEP.  04_experiments/independent_evaluation.py        13 scripts escritos a mano
  FASE 5  05_skills/sync_skills.py                        copia el auditor dentro de las Skills
  TESTS   08_tests/                                       pytest

Si un paso falla, el programa se detiene y muestra cuál fue.
"""

# %% PASO 0 · Configuración
# ------------------------------------------------------------------------------
import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent


# %% PASO 1 · Ejecutor de un paso
# ------------------------------------------------------------------------------
# LÓGICA: lanza cada script con el MISMO intérprete de Python, desde la raíz del proyecto y con
#   UTF-8 forzado (tildes correctas en Windows). Si el script falla, se detiene todo: ningún
#   resultado posterior se calcula sobre un paso roto.
# ------------------------------------------------------------------------------
def correr(titulo, *argumentos):
    print(f"\n{'=' * 78}\n{titulo}\n{'=' * 78}", flush=True)
    inicio = time.time()
    entorno = {**os.environ, "PYTHONUTF8": "1"}
    resultado = subprocess.run([sys.executable, *argumentos], cwd=RAIZ, env=entorno)
    if resultado.returncode != 0:
        sys.exit(f"\n✗ Falló: {titulo} (código {resultado.returncode})")
    print(f"✓ {titulo} ({time.time() - inicio:.0f} s)", flush=True)


# %% PASO 2 · Orquestación del protocolo
# ------------------------------------------------------------------------------
# En modo --rapido el benchmark escribe en results_rapido/ (ignorada por git) y se omiten los
# análisis que necesitan el benchmark completo.
# ------------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rapido = "--rapido" in sys.argv
    filtro = ["--datasets", "D1", "D4", "--familias", "RF", "LIN"] if rapido else []
    salida = ["--salida", str(RAIZ / "04_experiments" / "results_rapido")] if rapido else []

    correr("FASE 0 · Datos", "01_data/sklearn_loader.py")
    correr("FASE 1 · Banco de control (96 pares)", "02_pipelines/generate_controlled_pairs.py")
    correr("FASES 2-4 · Auditoría, ΔOpt y reparación", "04_experiments/run_benchmark.py", *filtro, *salida)
    if not rapido:
        correr("EDA · Análisis exploratorio", "04_experiments/eda_datasets.py")
        correr("ESTADÍSTICA · Familias de modelos", "04_experiments/statistical_analysis.py")
        correr("SENSIBILIDAD · Ruido vs. fuga por selección", "04_experiments/noise_sensitivity.py")
        correr("DEMO · Ejemplo visual de fuga (R3)", "04_experiments/leakage_demo.py")
        correr("LÍNEA BASE · Regla simple", "04_experiments/baseline_simple_rule.py")
        correr("EVALUACIÓN INDEPENDIENTE · 13 scripts escritos a mano", "04_experiments/independent_evaluation.py")
    correr("FASE 5 · Sincronizar Skills", "05_skills/sync_skills.py")
    correr("TESTS", "-m", "pytest", "08_tests", "-q")
    print("\n✓ Todo listo. Para la plataforma web:  streamlit run 06_web_app/app.py")


if __name__ == "__main__":
    main()
