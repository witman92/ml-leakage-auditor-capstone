"""
generar_con_llm.py  ·  PASO 1 (Conjunto A): pide a los LLM que escriban los pipelines.

⚠ ESTADO: escrito y revisado, pero NO ejecutado contra ninguna API en este proyecto (requiere claves
  de pago). Los resultados del informe de Capstone I NO dependen de este script: se basan en el banco
  de control (02_pipelines/controlled_pairs). Queda listo para Capstone II.

USO (ejemplos; el nombre exacto del modelo lo elige usted, para no fijar versiones que caducan):
    set ANTHROPIC_API_KEY=...   &&  python generar_con_llm.py --proveedor anthropic --modelo <id-del-modelo>
    set OPENAI_API_KEY=...      &&  python generar_con_llm.py --proveedor openai    --modelo <id-del-modelo>

Escribe  llm_generated/<modelo>/<DATASET>_<FAMILIA>.py  (24 archivos por modelo: 6 datasets × 4 familias).
Luego audite todo con:  python 04_experiments/auditar_llm_generados.py
Registre en el informe la versión exacta del modelo y la fecha de generación (reproducibilidad).
"""
import argparse
import os
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "02_pipelines"))
from generar_pares_controlados import DATASETS, FAMILIAS  # noqa: E402

PLANTILLA = (Path(__file__).parent / "PROMPT_ESTANDAR.md").read_text(encoding="utf-8")
PROMPT = PLANTILLA.split("```")[1].strip()          # el bloque entre ``` del archivo .md

CARGAS = {"D1": "from sklearn.datasets import load_breast_cancer; X, y = load_breast_cancer(return_X_y=True)",
          "D2": "from sklearn.datasets import load_wine; X, y = load_wine(return_X_y=True)",
          "D3": "from sklearn.datasets import load_digits; X, y = load_digits(return_X_y=True)",
          "D4": "from sklearn.datasets import load_diabetes; X, y = load_diabetes(return_X_y=True)",
          "D5": "from sklearn.datasets import fetch_california_housing; X, y = fetch_california_housing(return_X_y=True)",
          "D6": "from sklearn.datasets import make_friedman1; X, y = make_friedman1(n_samples=1000, n_features=50, noise=1.0, random_state=42)"}
MODELO_TEXTO = {"RF": ("RandomForest", "RandomForest"), "GB": ("HistGradientBoosting", "HistGradientBoosting"),
                "LIN": ("LogisticRegression (clasificación) o Ridge (regresión)", "lineal regularizado"),
                "MLP": ("MLPClassifier (clasificación) o MLPRegressor (regresión)", "perceptrón multicapa")}


def pedir(proveedor, modelo, texto):
    """Envía el prompt y devuelve el texto de la respuesta (SDK importado solo si se usa)."""
    if proveedor == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(model=modelo, max_tokens=2000,
                                                  messages=[{"role": "user", "content": texto}])
        return r.content[0].text
    if proveedor == "openai":
        import openai
        r = openai.OpenAI().chat.completions.create(model=modelo, messages=[{"role": "user", "content": texto}])
        return r.choices[0].message.content
    raise SystemExit("Proveedor no soportado: use 'anthropic' u 'openai' (añadir otro es una función más).")


def extraer_codigo(respuesta):
    """Si el modelo respondió con ```python ... ```, devuelve solo el código."""
    bloque = re.search(r"```(?:python)?\n(.*?)```", respuesta, re.S)
    return (bloque.group(1) if bloque else respuesta).strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proveedor", required=True, choices=["anthropic", "openai"])
    ap.add_argument("--modelo", required=True)
    args = ap.parse_args()
    clave = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}[args.proveedor]
    if not os.environ.get(clave):
        raise SystemExit(f"Falta la variable de entorno {clave}.")
    salida = Path(__file__).parent / re.sub(r"[^\w.-]", "_", args.modelo)
    salida.mkdir(exist_ok=True)
    for ident, tarea in DATASETS.items():
        for familia in FAMILIAS:
            texto = (PROMPT.replace("{TAREA}", tarea).replace("{DATASET}", ident)
                     .replace("{CARGA}", CARGAS[ident]).replace("{MODELO}", MODELO_TEXTO[familia][0])
                     .replace("{METRICAS}", "ROC-AUC y F1 macro" if tarea == "clasificacion" else "R2 y RMSE"))
            codigo = extraer_codigo(pedir(args.proveedor, args.modelo, texto))
            (salida / f"{ident}_{familia}.py").write_text(codigo, encoding="utf-8")
            print("  escrito", salida.name, f"{ident}_{familia}.py")


if __name__ == "__main__":
    main()
