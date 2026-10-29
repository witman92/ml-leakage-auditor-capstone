"""
test_proyecto.py  ·  Pruebas automáticas del proyecto.      Ejecutar:  python -m pytest 08_pruebas -v

Qué verifican (en orden):
  1. Datos         : formas y reproducibilidad de los 6 datasets.
  2. Banco         : el generador produce 96 pares y todos son código Python válido.
  3. Auditor       : evaluación INDEPENDIENTE con 13 scripts escritos a mano (no generados por el
                     mismo programa que el auditor), incluyendo un caso que el AST debe fallar.
  4. Verificador   : el runtime SÍ atrapa el caso que el AST no ve (importación con alias).
  5. Refactorizador: repara los dos patrones y NO repara los que no entiende (los declara manuales).
  6. Skills        : copias sincronizadas con 03_auditor, SKILL.md válido y plantilla limpia.
  7. Plataforma web: la lógica y la app arrancan sin errores.
  8. Resultados    : si ya se corrió el benchmark, el código reparado rinde igual que el limpio.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
for carpeta in ("01_data", "02_pipelines", "03_auditor", "06_web"):
    sys.path.insert(0, str(RAIZ / carpeta))

import generar_pares_controlados as generador                    # noqa: E402
import logica                                                    # noqa: E402
from ast_leakage_detector import analizar_archivo, analizar_codigo  # noqa: E402
from pipeline_fixer import corregir_archivo, corregir_codigo     # noqa: E402
from runtime_verifier import ejecutar_con_verificador            # noqa: E402
from sklearn_loader import DATASETS, cargar_dataset              # noqa: E402

CASOS = RAIZ / "08_pruebas" / "casos_manuales"
ESPERADO = json.loads((CASOS / "esperado.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 1. datos
@pytest.mark.parametrize("ident", list(DATASETS))
def test_datos_reproducibles_y_con_faltantes(ident):
    X1, y1 = cargar_dataset(ident)
    X2, _ = cargar_dataset(ident)
    assert np.array_equal(X1, X2)                          # misma semilla -> mismos datos
    assert not np.isnan(X1).any()                          # "completo" no tiene faltantes
    Xf, _ = cargar_dataset(ident, "con_faltantes")
    assert 0.08 < np.isnan(Xf).mean() < 0.12               # ~10 %
    assert len(y1) == X1.shape[0]


# ------------------------------------------------------------------ 2. banco de control
def test_generador_produce_96_pares_validos(tmp_path):
    assert generador.generar_todos(tmp_path) == 192
    for archivo in tmp_path.glob("*.py"):
        ast.parse(archivo.read_text(encoding="utf-8"))     # sintaxis válida
    assert len(list(tmp_path.glob("*_leaky.py"))) == len(list(tmp_path.glob("*_clean.py"))) == 96


# ------------------------------------------------------------------ 3. auditor (evaluación independiente)
@pytest.mark.parametrize("nombre", sorted(ESPERADO))
def test_auditor_casos_manuales(nombre):
    detectado = analizar_archivo(CASOS / nombre)["estado"] == "FUGA_DETECTADA"
    assert detectado == ESPERADO[nombre]["ast_debe_detectar"], ESPERADO[nombre]["descripcion"]


def test_auditor_no_marca_limpios():
    limpios = [n for n, v in ESPERADO.items() if not v["fuga"]]
    assert limpios and all(analizar_archivo(CASOS / n)["estado"] == "LIMPIO" for n in limpios)


def test_auditor_error_de_sintaxis():
    assert analizar_codigo("def roto(:\n")["estado"] == "ERROR_SINTAXIS"


# ------------------------------------------------------------------ 4. verificador en ejecución
def test_runtime_atrapa_lo_que_el_ast_no_ve():
    caso = CASOS / "09_fuga_con_alias_de_import.py"
    assert analizar_archivo(caso)["estado"] == "LIMPIO"           # el AST se equivoca (limitación conocida)
    assert ejecutar_con_verificador(caso)["fuga"] is True          # el runtime lo corrige


def test_runtime_no_marca_pipeline_limpio():
    r = ejecutar_con_verificador(CASOS / "04_limpio_make_pipeline.py")
    assert r["fuga"] is False and r["max_filas_ajuste"] < r["n_total"]


def test_runtime_restaura_metodos_originales():
    from sklearn.preprocessing import StandardScaler
    antes = StandardScaler.fit
    ejecutar_con_verificador(CASOS / "11_limpio_sin_transformadores.py")
    assert StandardScaler.fit is antes                             # no queda ningún parche


# ------------------------------------------------------------------ 5. refactorizador
@pytest.mark.parametrize("nombre", ["01_fuga_scaler_antes_de_split.py", "03_fuga_selectkbest_cv.py",
                                    "12_fuga_reasigna_la_misma_variable.py"])
def test_fixer_repara_patrones_conocidos(nombre):
    r = corregir_archivo(CASOS / nombre)
    assert r["reparado"] and r["auditoria_final"]["estado"] == "LIMPIO"
    ast.parse(r["codigo_corregido"])


def test_fixer_declara_lo_que_no_sabe_reparar():
    r = corregir_archivo(CASOS / "05_fuga_ajuste_con_datos_de_prueba.py")
    assert not r["reparado"] and r["requiere_revision_manual"]


def test_fixer_no_toca_codigo_limpio():
    r = corregir_archivo(CASOS / "04_limpio_make_pipeline.py")
    assert not r["reparado"] and not r["requiere_revision_manual"]


# ------------------------------------------------------------------ 6. skills
ARCHIVOS_SKILL = ["ast_leakage_detector.py", "pipeline_fixer.py", "run_ast_audit.py"]


@pytest.mark.parametrize("skill", ["audit-ml-leakage", "create-leakage-free-pipeline"])
def test_skills_sincronizadas_y_con_frontmatter(skill):
    for archivo in ARCHIVOS_SKILL:
        original = (RAIZ / "03_auditor" / archivo).read_text(encoding="utf-8")
        copia = (RAIZ / "05_skills" / skill / "scripts" / archivo).read_text(encoding="utf-8")
        assert original == copia, f"{skill}/{archivo} desactualizado: ejecute 05_skills/sincronizar_skills.py"
    texto = (RAIZ / "05_skills" / skill / "SKILL.md").read_text(encoding="utf-8")
    assert texto.startswith("---") and f"name: {skill}" in texto and "description:" in texto


def test_plantilla_segura_esta_limpia():
    plantilla = RAIZ / "05_skills" / "create-leakage-free-pipeline" / "templates" / "safe_sklearn_template.py"
    assert analizar_archivo(plantilla)["estado"] == "LIMPIO"


# ------------------------------------------------------------------ 7. plataforma web
def test_web_logica():
    codigo = (CASOS / "03_fuga_selectkbest_cv.py").read_text(encoding="utf-8")
    informe = logica.auditar(codigo)
    assert informe["estado"] == "FUGA_DETECTADA"
    html = logica.html_codigo_resaltado(codigo, {h["linea"] for h in informe["hallazgos"]})
    assert "background:#ffd6d6" in html
    reparacion = logica.reparar(codigo)
    assert reparacion["reparado"] and "Pipeline" in logica.diff_unificado(codigo, reparacion["codigo_corregido"])


def test_web_html_escapa_codigo_malicioso():
    html = logica.html_codigo_resaltado("x = '<script>alert(1)</script>'", set())
    assert "<script>" not in html


def test_web_app_arranca_sin_errores():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(RAIZ / "06_web" / "app.py"), default_timeout=60).run()
    assert not at.exception


# ------------------------------------------------------------------ 8. resultados del benchmark
def test_reparado_rinde_igual_que_limpio():
    ruta = RAIZ / "04_experiments" / "results" / "delta_optimism_pairs.csv"
    if not ruta.exists():
        pytest.skip("aún no se corrió run_benchmark.py")
    d = pd.read_csv(ruta)["diff_reparado_vs_clean"].dropna()
    assert len(d) > 0 and d.abs().max() < 1e-9
