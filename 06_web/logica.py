"""
logica.py  ·  Funciones de la plataforma web, SEPARADAS de la interfaz.

¿Por qué separarlas? Para poder probarlas con pytest sin abrir un navegador, y para que app.py
quede corto y fácil de leer (solo dibuja pantallas; aquí vive el "cerebro").
La plataforma reutiliza EXACTAMENTE el mismo auditor y refactorizador que las Skills.
"""
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
CARPETA_EJEMPLOS = RAIZ / "08_pruebas" / "casos_manuales"
CARPETA_PARES = RAIZ / "02_pipelines" / "controlled_pairs"


def auditar(codigo):
    """Audita el texto de un script (misma función que usa la Skill)."""
    return analizar_codigo(codigo)


def reparar(codigo):
    """Intenta la reparación automática. Devuelve el dict de pipeline_fixer.corregir_codigo."""
    return corregir_codigo(codigo)


def diff_unificado(original, nuevo):
    """Diferencias línea a línea entre dos versiones, en formato 'diff' clásico."""
    return "".join(difflib.unified_diff(
        original.splitlines(keepends=True), nuevo.splitlines(keepends=True),
        fromfile="original (con fuga)", tofile="corregido (sin fuga)"))


def html_codigo_resaltado(codigo, lineas_con_fuga):
    """Código con número de línea; las líneas con fuga salen en rojo. Devuelve HTML seguro."""
    filas = []
    for i, texto in enumerate(codigo.splitlines(), start=1):
        estilo = "background:#ffd6d6;" if i in lineas_con_fuga else ""
        filas.append(f'<div style="{estilo}white-space:pre;font-family:monospace;font-size:13px;">'
                     f'<span style="color:#888;">{i:>3} </span>{html.escape(texto)}</div>')
    return '<div style="border:1px solid #ddd;padding:6px;overflow-x:auto;">' + "".join(filas) + "</div>"


def lista_de_ejemplos():
    """{nombre visible: ruta} con los casos manuales de prueba (sirven de demostración)."""
    return {p.stem: p for p in sorted(CARPETA_EJEMPLOS.glob("*.py"))}


def cargar_resultados():
    """Lee los CSV del experimento. Devuelve None si aún no se ejecutó run_benchmark.py."""
    archivos = {"confusion": "auditor_confusion_matrix.csv", "por_tipo": "auditor_por_tipo.csv",
                "pares": "delta_optimism_pairs.csv", "tests": "tests_por_familia.csv",
                "friedman": "friedman_familias.csv", "eda": "eda_datasets.csv",
                "medias": "delta_opt_por_familia_fuga.csv"}
    if not (RESULTADOS / archivos["pares"]).exists():
        return None
    return {k: pd.read_csv(RESULTADOS / f) for k, f in archivos.items() if (RESULTADOS / f).exists()}


def par_controlado(pares, dataset, familia, fuga, metrica):
    """Fila de ΔOpt de un par concreto (o None)."""
    sel = pares[(pares["dataset"] == dataset) & (pares["familia"] == familia)
                & (pares["fuga"] == fuga) & (pares["metrica"] == metrica)]
    return None if sel.empty else sel.iloc[0]
