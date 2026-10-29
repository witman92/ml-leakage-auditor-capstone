# ==============================================================================
# app_logic.py · Lógica de la plataforma web, SEPARADA de la interfaz Streamlit
# ==============================================================================
"""
Funciones puras de la plataforma web (se prueban con pytest sin abrir un navegador).

app.py solo dibuja pantallas; aquí vive el "cerebro". La plataforma reutiliza EXACTAMENTE el mismo
auditor y refactorizador que las Skills (carpeta 03_auditor): una sola fuente de verdad.
"""

# %% PASO 0 · Configuración e importaciones
# ------------------------------------------------------------------------------
# QUÉ HACE   : agrega 03_auditor al path y fija las rutas de resultados y ejemplos.
# SEGURIDAD  : MAX_CARACTERES limita el tamaño del código aceptado (protege al servidor de
#              entradas enormes). Ninguna función de este módulo EJECUTA el código del usuario.
# ------------------------------------------------------------------------------
import difflib
import html
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "03_auditor"))
from ast_leakage_detector import analizar_codigo   # noqa: E402
from pipeline_fixer import corregir_codigo         # noqa: E402

RESULTADOS = RAIZ / "04_experiments" / "results"
CARPETA_EJEMPLOS = RAIZ / "08_tests" / "handwritten_cases"
CARPETA_PARES = RAIZ / "02_pipelines" / "controlled_pairs"
MAX_CARACTERES = 200_000


# %% PASO 1 · Validación de la entrada (medida de seguridad concreta)
# ------------------------------------------------------------------------------
# LÓGICA: rechaza entradas vacías o mayores que MAX_CARACTERES con un mensaje claro.
# DEVUELVE: (ok, mensaje). Si ok es False, la app muestra el mensaje y no audita.
# ------------------------------------------------------------------------------
def validar_entrada(codigo):
    if not codigo or not codigo.strip():
        return False, "Pega, sube o elige un script para auditarlo."
    if len(codigo) > MAX_CARACTERES:
        return False, (f"El código tiene {len(codigo):,} caracteres y el límite es {MAX_CARACTERES:,}. "
                       "Audite el archivo por partes o use la Skill desde la línea de comandos.")
    return True, ""


# %% PASO 2 · Auditoría y reparación (delegan en 03_auditor)
# ------------------------------------------------------------------------------
# auditar : análisis AST estático (reglas R-Split, R-CV, R-Test). No ejecuta nada.
# reparar : refactorizador determinista (patrones CV y Split) + re-auditoría del resultado.
# ------------------------------------------------------------------------------
def auditar(codigo):
    """Audita el texto de un script (misma función que usa la Skill)."""
    return analizar_codigo(codigo)


def reparar(codigo):
    """Intenta la reparación automática. Devuelve el dict de pipeline_fixer.corregir_codigo."""
    return corregir_codigo(codigo)


# %% PASO 3 · Presentación: diff y código resaltado
# ------------------------------------------------------------------------------
# diff_unificado       : diferencias línea a línea (formato diff clásico) entre original y reparado.
# html_codigo_resaltado: código numerado con las líneas con fuga en rojo. SEGURIDAD: todo el texto
#                        del usuario pasa por html.escape, así un "<script>" pegado no se ejecuta
#                        en el navegador (lo verifica un test).
# ------------------------------------------------------------------------------
def diff_unificado(original, nuevo):
    return "".join(difflib.unified_diff(
        original.splitlines(keepends=True), nuevo.splitlines(keepends=True),
        fromfile="original (con fuga)", tofile="corregido (sin fuga)"))


def html_codigo_resaltado(codigo, lineas_con_fuga):
    filas = []
    for i, texto in enumerate(codigo.splitlines(), start=1):
        estilo = "background:#ffd6d6;" if i in lineas_con_fuga else ""
        filas.append(f'<div style="{estilo}white-space:pre;font-family:monospace;font-size:13px;">'
                     f'<span style="color:#888;">{i:>3} </span>{html.escape(texto)}</div>')
    return '<div style="border:1px solid #ddd;padding:6px;overflow-x:auto;">' + "".join(filas) + "</div>"


# %% PASO 4 · Datos para las pestañas de resultados
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: los CSV que escriben run_benchmark.py, statistical_analysis.py y
#   eda_datasets.py en 04_experiments/results/. La app solo los LEE; si faltan, devuelve None y la
#   interfaz pide ejecutar el benchmark.
# ------------------------------------------------------------------------------
def lista_de_ejemplos():
    """{nombre visible: ruta} con los casos escritos a mano (sirven de demostración)."""
    return {p.stem: p for p in sorted(CARPETA_EJEMPLOS.glob("*.py"))}


def cargar_resultados():
    """Lee los CSV del experimento. Devuelve None si aún no se ejecutó run_benchmark.py."""
    archivos = {"confusion": "auditor_confusion_matrix.csv", "por_tipo": "auditor_per_leak_type.csv",
                "pares": "delta_optimism_pairs.csv", "tests": "tests_by_family.csv",
                "friedman": "friedman_families.csv", "eda": "eda_datasets.csv",
                "medias": "delta_opt_by_family_leak.csv"}
    if not (RESULTADOS / archivos["pares"]).exists():
        return None
    return {k: pd.read_csv(RESULTADOS / f) for k, f in archivos.items() if (RESULTADOS / f).exists()}


def par_controlado(pares, dataset, familia, fuga, metrica):
    """Fila de ΔOpt de un par concreto (o None)."""
    sel = pares[(pares["dataset"] == dataset) & (pares["familia"] == familia)
                & (pares["fuga"] == fuga) & (pares["metrica"] == metrica)]
    return None if sel.empty else sel.iloc[0]
