# ==============================================================================
# run_ast_audit.py · Herramienta de línea de comandos del auditor (la invocan las Skills y la CI)
# ==============================================================================
"""
Audita (y opcionalmente repara) un script de scikit-learn sin ejecutarlo.

    python run_ast_audit.py mi_pipeline.py                 # informe legible
    python run_ast_audit.py mi_pipeline.py --json          # salida JSON (para agentes y CI)
    python run_ast_audit.py mi_pipeline.py --fix           # intenta reparar -> mi_pipeline_sin_fuga.py
    python run_ast_audit.py mi_pipeline.py --fix salida.py # idem, con nombre de salida elegido

CÓDIGOS DE SALIDA (para usar en scripts y en integración continua)
    0 = limpio (o reparado con éxito)     1 = fuga detectada (y no reparada)     2 = error de lectura

Este archivo es la ÚNICA fuente: 05_skills/sync_skills.py lo copia dentro de cada Skill junto con
ast_leakage_detector.py y pipeline_fixer.py, para que cada Skill sea autocontenida.
"""

# %% PASO 0 · Importaciones: auditor y refactorizador hermanos
# ------------------------------------------------------------------------------
# QUÉ HACE   : agrega la carpeta del propio archivo al path para importar ast_leakage_detector y
#              pipeline_fixer; así funciona igual en 03_auditor/ que dentro de una Skill copiada.
# ------------------------------------------------------------------------------
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # importar los módulos hermanos
from ast_leakage_detector import analizar_archivo          # noqa: E402
from pipeline_fixer import corregir_archivo                # noqa: E402


# %% PASO 1 · Informe legible para personas (con la advertencia de alcance)
# ------------------------------------------------------------------------------
# QUÉ HACE   : imprime estado, resumen y, por hallazgo, línea, tipo/regla/confianza y el código.
# ------------------------------------------------------------------------------
# ALCANCE se muestra siempre que el veredicto es LIMPIO (y como clave "alcance" en --json):
# evita la falsa seguridad de leer "0 fugas" como "ninguna fuga posible".
ALCANCE = ("Alcance: significa «sin evidencia de las reglas R-Split, R-CV y R-Test», "
           "no «sin ningún tipo de fuga».")


def _imprimir(r, ruta):
    print(f"Archivo : {ruta}")
    print(f"Estado  : {r['estado']}  ({r['resumen']})")
    for h in r["hallazgos"]:
        print(f"  - línea {h['linea']:>3} [{h['tipo']}/{h['regla']}/{h['confianza']}] {h['mensaje']}")
        print(f"      > {h['codigo']}")
    if r["estado"] == "LIMPIO":
        print(ALCANCE)


# %% PASO 2 · Flujo principal: leer → auditar → (reparar) → informar → código de salida
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: la ruta del .py recibida en la línea de comandos.
# LÓGICA     : 1) valida argumentos y existencia del archivo (código 2 si falla);  2) audita;
#              3) con --fix y fuga detectada, intenta reparar y escribe <archivo>_sin_fuga.py (o el
#              nombre indicado);  4) imprime texto o JSON (--json, pensado para agentes y CI).
# SALIDA     : 0 = limpio o reparado; 1 = fuga sin reparar; 2 = error de lectura o sintaxis.
# INTERPRETACIÓN: un 1 en CI debe bloquear la integración hasta que alguien revise el pipeline.
# ------------------------------------------------------------------------------
def main(argv):
    for flujo in (sys.stdout, sys.stderr):          # tildes correctas en la consola de Windows
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8")
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    ruta = Path(argv[1])
    if not ruta.exists():
        print(f"No existe el archivo: {ruta}", file=sys.stderr)
        return 2
    como_json = "--json" in argv
    arreglar = "--fix" in argv

    informe = analizar_archivo(ruta)
    if informe["estado"] == "ERROR_SINTAXIS":
        print(informe["resumen"], file=sys.stderr)
        return 2

    reparacion = None
    if arreglar and informe["estado"] == "FUGA_DETECTADA":
        i = argv.index("--fix")
        salida = (Path(argv[i + 1]) if len(argv) > i + 1 and not argv[i + 1].startswith("--")
                  else ruta.with_name(ruta.stem + "_sin_fuga.py"))
        reparacion = corregir_archivo(ruta, salida)
        reparacion["archivo_salida"] = str(salida) if reparacion["reparado"] else None

    if como_json:
        print(json.dumps({"auditoria": informe, "alcance": ALCANCE,
                          "reparacion": ({k: v for k, v in reparacion.items() if k != "auditoria_final"}
                                         | {"auditoria_final": reparacion["auditoria_final"]["estado"]})
                          if reparacion else None}, ensure_ascii=False, indent=2))
    else:
        _imprimir(informe, ruta)
        if reparacion:
            if reparacion["reparado"]:
                print(f"\nREPARADO -> {reparacion['archivo_salida']}  ({reparacion['mensaje']})")
                print(f"Auditoría del archivo nuevo: {reparacion['auditoria_final']['resumen']}")
            else:
                print(f"\nNO SE PUDO REPARAR AUTOMÁTICAMENTE: {reparacion['mensaje']}")
                print("Requiere revisión manual: encierre los transformadores en sklearn.pipeline.Pipeline.")

    if informe["estado"] == "LIMPIO" or (reparacion and reparacion["reparado"]):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
