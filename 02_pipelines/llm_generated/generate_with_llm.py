# ==============================================================================
# generate_with_llm.py · PASO 1 (Conjunto A): los LLM escriben los pipelines (sujetos de estudio)
# ==============================================================================
"""
Pide a un LLM generador que escriba los 24 pipelines del estudio (6 datasets × 4 familias) con el
MISMO prompt estándar (standard_prompt.md) y guarda cada respuesta como un archivo .py.

ESTADO: escrito y revisado, pero NO ejecutado contra ninguna API en Capstone I (requiere claves de
pago). Los resultados del informe NO dependen de este script: se basan en el banco de control
(02_pipelines/controlled_pairs). Queda listo para Capstone II.

Uso (el identificador exacto del modelo lo elige usted, para no fijar versiones que caducan):
    set ANTHROPIC_API_KEY=...  &&  python 02_pipelines/llm_generated/generate_with_llm.py --proveedor anthropic --modelo <id>
    set OPENAI_API_KEY=...     &&  python 02_pipelines/llm_generated/generate_with_llm.py --proveedor openai    --modelo <id>

Salida: 02_pipelines/llm_generated/<modelo>/<DATASET>_<FAMILIA>.py  (24 archivos por modelo).
Después: python 04_experiments/audit_llm_generated.py   (auditoría estática, sin ejecutar el código).
"""

# %% PASO 0 · Separación de roles: generador ≠ auditor (observación R2)
# ------------------------------------------------------------------------------
# QUÉ HACE   : importa lo necesario y documenta el rol de este script en el diseño.
# POR QUÉ    : los LLM son SOLO sujetos de estudio (escriben código). El instrumento de medición es
#              el auditor determinista de 03_auditor (AST + verificación en ejecución), que nunca
#              consulta a un LLM. Así se evita el sesgo de auto-preferencia de un LLM que se juzga
#              a sí mismo y la variabilidad no reproducible de un "LLM juez".
# REPRODUCIBILIDAD: registre en el informe el id exacto del modelo y la fecha de generación.
# ------------------------------------------------------------------------------
import argparse
import os
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "02_pipelines"))
from generate_controlled_pairs import DATASETS, FAMILIAS  # noqa: E402  (mismos factores que el banco)


# %% PASO 1 · Prompt estándar y textos que se sustituyen
# ------------------------------------------------------------------------------
# ORIGEN     : el prompt vive en standard_prompt.md (bloque entre ```), versionado en el repositorio.
# LÓGICA     : el texto es idéntico para todos los modelos; solo se reemplazan {TAREA}, {DATASET},
#              {CARGA}, {MODELO} y {METRICAS}. El prompt NO menciona la fuga: se mide la tendencia
#              espontánea del modelo a cometerla.
# CARGAS     : la línea de código con que cada dataset se carga desde sklearn.datasets (nota: aquí
#              california_housing va completo; el submuestreo es una decisión del banco de control).
# ------------------------------------------------------------------------------
PLANTILLA = (Path(__file__).parent / "standard_prompt.md").read_text(encoding="utf-8")
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


# %% PASO 2 · Llamada a la API del proveedor
# ------------------------------------------------------------------------------
# QUÉ HACE   : envía el prompt y devuelve el texto de la respuesta.
# LÓGICA     : el SDK se importa solo si se usa ese proveedor (el proyecto no exige instalarlos).
#              Agregar otro proveedor es agregar una rama más.
# ------------------------------------------------------------------------------
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


# %% PASO 3 · Extracción del código de la respuesta
# ------------------------------------------------------------------------------
# QUÉ HACE   : si el modelo respondió con un bloque ```python ... ```, conserva solo el código;
#              si respondió texto plano, lo guarda tal cual (la auditoría dirá si es Python válido).
# ------------------------------------------------------------------------------
def extraer_codigo(respuesta):
    """Si el modelo respondió con ```python ... ```, devuelve solo el código."""
    bloque = re.search(r"```(?:python)?\n(.*?)```", respuesta, re.S)
    return (bloque.group(1) if bloque else respuesta).strip() + "\n"


# %% PASO 4 · Generación de los 24 scripts por modelo
# ------------------------------------------------------------------------------
# QUÉ HACE   : recorre 6 datasets × 4 familias, completa el prompt, llama al modelo y escribe
#              <DATASET>_<FAMILIA>.py en una carpeta con el nombre del modelo.
# SEGURIDAD  : la clave se lee de una variable de entorno (nunca se escribe en el código) y los
#              scripts generados NO se ejecutan aquí: son código no confiable.
# ------------------------------------------------------------------------------
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
