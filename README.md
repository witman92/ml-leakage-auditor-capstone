# Auditoría y prevención de fuga de datos en pipelines de ML generados por LLM
**Capstone I · Magíster en Data Science · UDLA** — Autor: Witman Zavala Muñoz · Profesor Guía: Francisco Pérez

> **La idea en una frase.** Un pipeline de Machine Learning puede ajustar su preprocesamiento (escalado, imputación,
> selección de variables) con *todas* las filas, incluidas las de prueba. El código no falla ni avisa, pero la métrica
> sale inflada. Este proyecto **detecta** esa fuga leyendo el código, **mide cuánto infla** la métrica y la **repara**.

## Cómo abrirlo (VS Code)
1. `File > Open Workspace from File…` → elegir **`capstone.code-workspace`** (todo el proyecto, con tareas y depuración listas).
2. Terminal: `pip install -r requirements.txt`
3. **Un comando para reproducir todo:** `python ejecutar_todo.py` (≈ 5-8 min) — o `--rapido` (≈ 30 s).
   También: `Terminal > Run Task…` y elegir el paso.

## Mapa del proyecto (se lee en orden: cada carpeta es un paso)

| Carpeta | Paso | Qué contiene | Archivo clave |
|---|---|---|---|
| `01_data/` | 0 | Carga de los 6 datasets de scikit-learn (variantes sin y con valores faltantes, columnas de ruido) | `sklearn_loader.py` |
| `02_pipelines/` | 1 | Banco de control: **96 pares** (6 datasets × 4 familias × 4 fugas) con y sin fuga = 192 scripts. Prompt y generador para el Conjunto A (LLM, sin ejecutar) | `generar_pares_controlados.py` |
| `03_auditor/` | 2 | **El auditor**: análisis estático (AST), verificador en ejecución, refactorizador y línea de comandos | `ast_leakage_detector.py` |
| `04_experiments/` | 3-4 | Benchmark, EDA, estadística y sensibilidad; `results/` guarda todas las tablas y figuras | `run_benchmark.py` |
| `05_skills/` | 5 | **Producto 1**: dos Agent Skills autocontenidas | `audit-ml-leakage/SKILL.md` |
| `06_web/` | 5 | **Producto 2**: plataforma web (alternativa a las Skills) | `app.py` |
| `07_informe/` | — | Informe de entrega final (rúbrica v2), checklist de la rúbrica y registro de decisiones | `INFORME_EF.md` |
| `08_pruebas/` | — | Pruebas automáticas, con 13 casos escritos a mano como evaluación independiente | `test_proyecto.py` |

## Los 5 pasos del experimento

```
PASO 1  Generar 96 pares leaky/clean (verdad conocida por construcción)      02_pipelines/
PASO 2  Auditar cada script: AST (lee el código) + runtime (cuenta filas en .fit)   03_auditor/
PASO 3  Comparar con la verdad conocida → precisión, recall, F1, VPN              04_experiments/run_benchmark.py
PASO 4  ΔOpt = métrica(con fuga) − métrica(sin fuga), mismos pliegues; tests por familia
PASO 5  Empaquetar el auditor como 2 Skills y como plataforma web
```

**Métrica central — Sesgo de Optimismo:** `ΔOpt = M_leaky − M_clean`, con el mismo dataset, los mismos 5 pliegues
(semilla 42), el mismo modelo y los mismos hiperparámetros. Como solo cambia si el preprocesamiento va fuera o dentro
de un `Pipeline`, toda diferencia se debe a la fuga. Si no hay fuga, ΔOpt = 0 por identidad.

## Los productos

**Skills** (`05_skills/`): copie la carpeta de la Skill a `~/.claude/skills/` (o al directorio de Skills de su agente).
- `audit-ml-leakage` → audita y repara scripts existentes (`python scripts/run_ast_audit.py archivo.py --fix`).
- `create-leakage-free-pipeline` → plantilla y reglas para crear pipelines sin fuga, con autoverificación.

**Plataforma web** (`06_web/`): `streamlit run 06_web/app.py` → pega o sube un `.py`, ve las líneas con fuga, descarga
el código corregido y consulta los resultados del experimento. Solo *lee* el código (nunca lo ejecuta), por eso es seguro
publicarla. Use la web si no trabaja con un agente de programación; use las Skills si sí.

## Limitaciones que conviene decir antes de que las pregunten
- El auditor es **estático**: sigue variables por nombre; no ve alias de importación (`import … as SS`), `exec`, ni fugas
  que dependen de valores (duplicados). El verificador en ejecución cubre parte de eso. Medidas en `07_informe/INFORME_EF.md`.
- «0 fugas detectadas» significa *sin evidencia de las reglas R-Split, R-CV y R-Test*, no *sin ningún tipo de fuga*.
- El banco de 192 scripts lo generó el mismo equipo que escribió el auditor (evaluación circular); por eso existe la
  prueba independiente con 13 scripts manuales.
- Los **pipelines escritos por LLM reales no se han generado todavía** (queda para Capstone II); los resultados actuales
  vienen del banco de control. Detalle y desviaciones: `07_informe/decision_log.md`.

## Reproducibilidad
Python 3.13, versiones fijas en `requirements.txt`, semilla única 42, sin internet salvo la primera descarga de
`california_housing` (scikit-learn la guarda en caché). Las pruebas: `python -m pytest 08_pruebas -q`.
