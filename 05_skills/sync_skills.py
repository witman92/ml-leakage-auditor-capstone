# ==============================================================================
# sync_skills.py · PASO 5 del experimento: empaquetar el auditor dentro de cada Skill
# ==============================================================================
"""
Copia los tres archivos del auditor (03_auditor/) dentro de la carpeta scripts/ de cada Skill.

Ejecución (cada vez que cambie el auditor):
    python 05_skills/sync_skills.py
"""

# %% PASO 0 · Por qué copiar en vez de importar
# ------------------------------------------------------------------------------
# QUÉ HACE   : define la fuente única y los destinos.
# POR QUÉ    : una Skill debe ser una carpeta autocontenida: el usuario lleva audit-ml-leakage/ a
#              ~/.claude/skills/ (o al directorio de Skills de su agente) y debe funcionar sin el
#              resto del repositorio. La FUENTE ÚNICA sigue siendo 03_auditor/.
# CONTROL    : 08_tests/test_project.py compara byte a byte las copias con el original y falla si
#              quedan desactualizadas (evita que la Skill audite con una versión vieja).
# ------------------------------------------------------------------------------
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVOS = ["ast_leakage_detector.py", "pipeline_fixer.py", "run_ast_audit.py"]
SKILLS = ["audit-ml-leakage", "create-leakage-free-pipeline"]


# %% PASO 1 · Sincronización
# ------------------------------------------------------------------------------
# LÓGICA     : para cada Skill crea scripts/ si no existe y sobrescribe los tres archivos.
# SALIDA     : una línea por Skill con los archivos copiados.
# ------------------------------------------------------------------------------
def sincronizar():
    for skill in SKILLS:
        destino = RAIZ / "05_skills" / skill / "scripts"
        destino.mkdir(parents=True, exist_ok=True)
        for archivo in ARCHIVOS:
            shutil.copyfile(RAIZ / "03_auditor" / archivo, destino / archivo)
        print(f"  {skill}/scripts/ <- {', '.join(ARCHIVOS)}")


if __name__ == "__main__":
    sincronizar()
