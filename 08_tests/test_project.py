# ==============================================================================
# test_project.py · Pruebas automáticas del proyecto (pytest)
# ==============================================================================
"""
Ejecutar:  python -m pytest 08_tests -v

Qué verifican (en orden):
  1. Datos          : formas y reproducibilidad de los 6 datasets.
  2. Banco          : el generador produce 96 pares y todos son código Python válido.
  3. Auditor        : evaluación INDEPENDIENTE con 13 scripts escritos a mano, incluido un caso que
                      el AST debe fallar (limitación conocida).
  4. Verificador    : el runtime SÍ atrapa el caso que el AST no ve (importación con alias).
  5. Refactorizador : repara los dos patrones y NO repara los que no entiende (los declara manuales).
  6. Skills         : copias sincronizadas con 03_auditor, SKILL.md válido y plantilla limpia.
  7. Plataforma web : la lógica, la validación de entrada y la app arrancan sin errores.
  8. Estadística    : ajuste de Holm e IC exacto verificados contra valores conocidos.
  9. Resultados     : si ya se corrió el benchmark, el código reparado rinde igual que el limpio.
"""

# %% PASO 0 · Configuración: rutas e importaciones del proyecto
# ------------------------------------------------------------------------------
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
for carpeta in ("01_data", "02_pipelines", "03_auditor", "04_experiments", "06_web_app"):
    sys.path.insert(0, str(RAIZ / carpeta))

import app_logic                                                 # noqa: E402
import generate_controlled_pairs as generador                    # noqa: E402
from ast_leakage_detector import analizar_archivo, analizar_codigo  # noqa: E402
from pipeline_fixer import corregir_archivo                      # noqa: E402
from runtime_verifier import ejecutar_con_verificador            # noqa: E402
from sklearn_loader import DATASETS, cargar_dataset              # noqa: E402

CASOS = RAIZ / "08_tests" / "handwritten_cases"
ESPERADO = json.loads((CASOS / "expected_labels.json").read_text(encoding="utf-8"))


# %% PASO 1 · Datos: reproducibilidad y variante con faltantes
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("ident", list(DATASETS))
def test_datos_reproducibles_y_con_faltantes(ident):
    X1, y1 = cargar_dataset(ident)
    X2, _ = cargar_dataset(ident)
    assert np.array_equal(X1, X2)                          # misma semilla -> mismos datos
    assert not np.isnan(X1).any()                          # "completo" no tiene faltantes
    Xf, _ = cargar_dataset(ident, "con_faltantes")
    assert 0.08 < np.isnan(Xf).mean() < 0.12               # ~10 %
    assert len(y1) == X1.shape[0]


# %% PASO 2 · Banco de control: 96 pares sintácticamente válidos
# ------------------------------------------------------------------------------
def test_generador_produce_96_pares_validos(tmp_path):
    assert generador.generar_todos(tmp_path) == 192
    for archivo in tmp_path.glob("*.py"):
        ast.parse(archivo.read_text(encoding="utf-8"))     # sintaxis válida
    assert len(list(tmp_path.glob("*_leaky.py"))) == len(list(tmp_path.glob("*_clean.py"))) == 96


# %% PASO 3 · Auditor AST: evaluación independiente con casos escritos a mano
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("nombre", sorted(ESPERADO))
def test_auditor_casos_manuales(nombre):
    detectado = analizar_archivo(CASOS / nombre)["estado"] == "FUGA_DETECTADA"
    assert detectado == ESPERADO[nombre]["ast_debe_detectar"], ESPERADO[nombre]["descripcion"]


def test_auditor_no_marca_limpios():
    limpios = [n for n, v in ESPERADO.items() if not v["fuga"]]
    assert limpios and all(analizar_archivo(CASOS / n)["estado"] == "LIMPIO" for n in limpios)


def test_auditor_error_de_sintaxis():
    assert analizar_codigo("def roto(:\n")["estado"] == "ERROR_SINTAXIS"


# %% PASO 4 · Verificador en ejecución
# ------------------------------------------------------------------------------
def test_runtime_atrapa_lo_que_el_ast_no_ve():
    caso = CASOS / "09_leaky_import_alias.py"
    assert analizar_archivo(caso)["estado"] == "LIMPIO"           # el AST se equivoca (limitación conocida)
    assert ejecutar_con_verificador(caso)["fuga"] is True          # el runtime lo corrige


def test_runtime_no_marca_pipeline_limpio():
    r = ejecutar_con_verificador(CASOS / "04_clean_make_pipeline.py")
    assert r["fuga"] is False and r["max_filas_ajuste"] < r["n_total"]


def test_runtime_restaura_metodos_originales():
    from sklearn.preprocessing import StandardScaler
    antes = StandardScaler.fit
    ejecutar_con_verificador(CASOS / "11_clean_no_transformers.py")
    assert StandardScaler.fit is antes                             # no queda ningún parche


# %% PASO 5 · Refactorizador
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("nombre", ["01_leaky_scaler_before_split.py", "03_leaky_selectkbest_cv.py",
                                    "12_leaky_reassigns_same_variable.py"])
def test_fixer_repara_patrones_conocidos(nombre):
    r = corregir_archivo(CASOS / nombre)
    assert r["reparado"] and r["auditoria_final"]["estado"] == "LIMPIO"
    ast.parse(r["codigo_corregido"])


def test_fixer_declara_lo_que_no_sabe_reparar():
    r = corregir_archivo(CASOS / "05_leaky_refit_on_test_data.py")
    assert not r["reparado"] and r["requiere_revision_manual"]


def test_fixer_no_toca_codigo_limpio():
    r = corregir_archivo(CASOS / "04_clean_make_pipeline.py")
    assert not r["reparado"] and not r["requiere_revision_manual"]


# %% PASO 6 · Skills autocontenidas
# ------------------------------------------------------------------------------
ARCHIVOS_SKILL = ["ast_leakage_detector.py", "pipeline_fixer.py", "run_ast_audit.py"]


@pytest.mark.parametrize("skill", ["audit-ml-leakage", "create-leakage-free-pipeline"])
def test_skills_sincronizadas_y_con_frontmatter(skill):
    for archivo in ARCHIVOS_SKILL:
        original = (RAIZ / "03_auditor" / archivo).read_text(encoding="utf-8")
        copia = (RAIZ / "05_skills" / skill / "scripts" / archivo).read_text(encoding="utf-8")
        assert original == copia, f"{skill}/{archivo} desactualizado: ejecute 05_skills/sync_skills.py"
    texto = (RAIZ / "05_skills" / skill / "SKILL.md").read_text(encoding="utf-8")
    assert texto.startswith("---") and f"name: {skill}" in texto and "description:" in texto


def test_plantilla_segura_esta_limpia():
    plantilla = RAIZ / "05_skills" / "create-leakage-free-pipeline" / "templates" / "safe_sklearn_template.py"
    assert analizar_archivo(plantilla)["estado"] == "LIMPIO"


# %% PASO 7 · Plataforma web
# ------------------------------------------------------------------------------
def test_web_logica():
    codigo = (CASOS / "03_leaky_selectkbest_cv.py").read_text(encoding="utf-8")
    informe = app_logic.auditar(codigo)
    assert informe["estado"] == "FUGA_DETECTADA"
    html = app_logic.html_codigo_resaltado(codigo, {h["linea"] for h in informe["hallazgos"]})
    assert "background:#ffd6d6" in html
    reparacion = app_logic.reparar(codigo)
    assert reparacion["reparado"] and "Pipeline" in app_logic.diff_unificado(codigo, reparacion["codigo_corregido"])


def test_web_html_escapa_codigo_malicioso():
    html = app_logic.html_codigo_resaltado("x = '<script>alert(1)</script>'", set())
    assert "<script>" not in html


def test_web_valida_tamano_de_entrada():
    assert app_logic.validar_entrada("x = 1\n")[0] is True
    assert app_logic.validar_entrada("   ")[0] is False
    ok, mensaje = app_logic.validar_entrada("x" * (app_logic.MAX_CARACTERES + 1))
    assert ok is False and "límite" in mensaje


def test_web_app_arranca_sin_errores():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(RAIZ / "06_web_app" / "app.py"), default_timeout=60).run()
    assert not at.exception


# %% PASO 8 · Estadística: valores de referencia conocidos
# ------------------------------------------------------------------------------
def test_holm_contra_valores_conocidos():
    from statistical_analysis import holm
    # p = [0.01, 0.04, 0.03] → ordenados 0.01·3 = 0.03; 0.03·2 = 0.06; 0.04·1 → max(0.06, 0.04) = 0.06
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert holm([0.9, 0.8])[0] == 1.0                              # recortado a 1


def test_clopper_pearson_extremos():
    from independent_evaluation import clopper_pearson
    inf, sup = clopper_pearson(9, 9)
    assert sup == 1.0 and abs(inf - 0.6637) < 1e-3                 # IC95 % exacto de 9/9


# %% PASO 9 · Resultados del benchmark (si ya se corrió)
# ------------------------------------------------------------------------------
def test_reparado_rinde_igual_que_limpio():
    ruta = RAIZ / "04_experiments" / "results" / "delta_optimism_pairs.csv"
    if not ruta.exists():
        pytest.skip("aún no se corrió run_benchmark.py")
    d = pd.read_csv(ruta)["diff_reparado_vs_clean"].dropna()
    assert len(d) > 0 and d.abs().max() < 1e-9
