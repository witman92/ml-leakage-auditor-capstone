"""
sincronizar_skills.py  ·  Copia el auditor dentro de cada Skill.

¿Por qué copiar? Una Skill debe ser una carpeta autocontenida: el usuario puede llevarse
`audit-ml-leakage/` a ~/.claude/skills/ (o a otro agente) y debe funcionar sin el resto del proyecto.
La FUENTE ÚNICA de los tres archivos está en 03_auditor/. Ejecute este script cada vez que cambie
el auditor (el test 08_pruebas/test_skills.py falla si las copias quedan desactualizadas).

    python 05_skills/sincronizar_skills.py
"""
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVOS = ["ast_leakage_detector.py", "pipeline_fixer.py", "run_ast_audit.py"]
SKILLS = ["audit-ml-leakage", "create-leakage-free-pipeline"]


def sincronizar():
    for skill in SKILLS:
        destino = RAIZ / "05_skills" / skill / "scripts"
        destino.mkdir(parents=True, exist_ok=True)
        for archivo in ARCHIVOS:
            shutil.copyfile(RAIZ / "03_auditor" / archivo, destino / archivo)
        print(f"  {skill}/scripts/ <- {', '.join(ARCHIVOS)}")


if __name__ == "__main__":
    sincronizar()
