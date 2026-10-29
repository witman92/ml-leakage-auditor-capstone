"""
ejecutar_todo.py  ·  Reproduce TODO el experimento con un solo comando.

    python ejecutar_todo.py            # completo (≈ 5-8 min)
    python ejecutar_todo.py --rapido   # 2 datasets × 2 familias (≈ 30 s), para probar que todo funciona

Cada paso es un script independiente que también se puede ejecutar y leer por separado:

  PASO 0  01_data/sklearn_loader.py               carga de los 6 datasets (prueba rápida)
  PASO 1  02_pipelines/generar_pares_controlados.py   escribe los 96 pares leaky/clean (192 scripts)
  PASO 2-4 04_experiments/run_benchmark.py        audita, verifica, calcula ΔOpt y repara
  EDA     04_experiments/eda_datasets.py          análisis exploratorio de los datasets
  ESTAD.  04_experiments/statistical_analysis.py  tests por familia, Friedman, figuras
  SENSIB. 04_experiments/sensibilidad_ruido.py    fuga por selección vs. cantidad de ruido
  PASO 5  05_skills/sincronizar_skills.py         copia el auditor dentro de las Skills
  TESTS   08_pruebas/                             pytest

Si un paso falla, el programa se detiene y muestra cuál fue.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent


def correr(titulo, *argumentos):
    """Ejecuta un script con el mismo Python que está corriendo este programa."""
    print(f"\n{'=' * 78}\n{titulo}\n{'=' * 78}", flush=True)
    inicio = time.time()
    entorno = {**os.environ, "PYTHONUTF8": "1"}      # tildes y símbolos correctos en la consola de Windows
    resultado = subprocess.run([sys.executable, *argumentos], cwd=RAIZ, env=entorno)
    if resultado.returncode != 0:
        sys.exit(f"\n✗ Falló: {titulo} (código {resultado.returncode})")
    print(f"✓ {titulo} ({time.time() - inicio:.0f} s)", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rapido = "--rapido" in sys.argv
    filtro = ["--datasets", "D1", "D4", "--familias", "RF", "LIN"] if rapido else []
    salida = ["--salida", str(RAIZ / "04_experiments" / "results_rapido")] if rapido else []

    correr("PASO 0 · Datos", "01_data/sklearn_loader.py")
    correr("PASO 1 · Banco de control (96 pares)", "02_pipelines/generar_pares_controlados.py")
    correr("PASOS 2-4 · Auditoría, ΔOpt y reparación", "04_experiments/run_benchmark.py", *filtro, *salida)
    if not rapido:   # los análisis siguientes leen results/ y necesitan el benchmark completo
        correr("EDA · Análisis exploratorio", "04_experiments/eda_datasets.py")
        correr("ESTADÍSTICA · Familias de modelos", "04_experiments/statistical_analysis.py")
        correr("SENSIBILIDAD · Ruido vs. fuga por selección", "04_experiments/sensibilidad_ruido.py")
    correr("PASO 5 · Sincronizar Skills", "05_skills/sincronizar_skills.py")
    correr("TESTS", "-m", "pytest", "08_pruebas", "-q")
    print("\n✓ Todo listo. Para la plataforma web:  streamlit run 06_web/app.py")


if __name__ == "__main__":
    main()
