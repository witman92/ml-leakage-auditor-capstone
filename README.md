# ml-leakage-auditor-capstone

**Auditoría y prevención de fuga de datos en pipelines de Machine Learning generados por LLM**
Capstone I · Magíster en Data Science · UDLA — Autor: Witman Zavala Muñoz · Profesor Guía: Francisco Pérez

> **La idea en una frase.** Un pipeline de Machine Learning puede ajustar su preprocesamiento (escalado, imputación,
> selección de variables) con *todas* las filas, incluidas las de prueba. El código no falla ni avisa, pero la métrica
> sale inflada. Este proyecto **detecta** esa fuga leyendo y ejecutando el código, **mide cuánto infla** la métrica
> (ΔOpt) y la **repara**. El auditor es determinista: nunca consulta a un LLM.

## Inicio rápido

```bash
pip install -r requirements.txt          # versiones fijas (Python 3.13)
python run_all.py                        # reproduce todo: fases 0-5, estadística, figuras y pruebas (≈ 5-8 min)
python run_all.py --rapido               # verificación rápida (≈ 30 s)
python 04_experiments/leakage_demo.py    # ejemplo visual de la fuga (Figura 1.2 del informe)
streamlit run 06_web_app/app.py          # plataforma web
python -m pytest 08_tests -q             # pruebas automáticas
```

En VS Code: `File > Open Workspace from File…` → `capstone.code-workspace` (una tarea por fase en `Terminal > Run Task…`).
Cada script está organizado en bloques `# %% PASO n` que se ejecutan uno a uno con *Run Cell* (o se pegan en Colab);
cada bloque explica el origen de los datos, la lógica, la métrica o gráfico que produce y cómo interpretarlo.

## Estructura del repositorio (cada carpeta es una fase del protocolo)

| Carpeta | Fase | Contenido | Archivo clave |
|---|---|---|---|
| `01_data/` | 0 | Carga determinista de 6 datasets nativos de scikit-learn (variantes con faltantes y columnas de ruido) | `sklearn_loader.py` |
| `02_pipelines/` | 1 | Banco de control: **96 pares** con y sin fuga (`controlled_pairs/`), versiones reparadas (`repaired_pairs/`) y generador del Conjunto A con LLM (`llm_generated/`, sin ejecutar) | `generate_controlled_pairs.py` |
| `03_auditor/` | 2 | Auditor: análisis estático (AST), verificador en ejecución, refactorizador y CLI | `ast_leakage_detector.py` |
| `04_experiments/` | 3–4 | Benchmark, EDA, estadística, sensibilidad al ruido, demo, línea base y evaluación independiente; `results/` guarda tablas, figuras y el manifiesto de ejecución | `run_benchmark.py` |
| `05_skills/` | 5 | **Producto 1**: dos Agent Skills autocontenidas | `audit-ml-leakage/SKILL.md` |
| `06_web_app/` | 5 | **Producto 2**: plataforma web en Streamlit (interfaz `app.py` + lógica `app_logic.py`) | `app.py` |
| `07_report/` | — | Informe LaTeX (plantilla UDLA), registro de decisiones y checklist de la rúbrica | `latex/capstone_report_ef.tex` |
| `08_tests/` | — | Pruebas automáticas y 13 casos escritos a mano (evaluación independiente) | `test_project.py` |

## Protocolo experimental en 5 fases

```
FASE 0  Datos: 6 datasets de sklearn + EDA (calidad, escalas, balance)            01_data/, eda_datasets.py
FASE 1  Generar 96 pares leaky/clean: verdad conocida por construcción            02_pipelines/generate_controlled_pairs.py
FASE 2  Auditar cada script: AST (lee el código) + runtime (cuenta filas en .fit)  03_auditor/
FASE 3  Eficacia del auditor: precisión, recall, VPN vs. verdad y vs. línea base   run_benchmark.py, independent_evaluation.py
FASE 4  ΔOpt = métrica(con fuga) − métrica(sin fuga), mismos pliegues; pruebas    statistical_analysis.py, noise_sensitivity.py
FASE 5  Empaquetar el auditor como 2 Skills y como plataforma web                  05_skills/, 06_web_app/
```

**Métrica central — Sesgo de Optimismo Inducido por Fuga:** `ΔOpt = M_leaky − M_clean`, con el mismo dataset, los
mismos 5 pliegues (semilla 42), el mismo modelo y los mismos hiperparámetros. Solo cambia si el preprocesamiento va fuera
o dentro de un `Pipeline`, así que toda diferencia se debe a la fuga. Si no hay fuga, ΔOpt = 0 por identidad.

## Arquitectura: quién genera y quién audita

| Componente | Rol | ¿Usa LLM? |
|---|---|---|
| Conjunto A (`02_pipelines/llm_generated/`) | Sujetos de estudio: pipelines escritos por LLM (Capstone II) | Sí |
| Conjunto B (`02_pipelines/controlled_pairs/`) | Oráculo programático: verdad conocida por construcción | No |
| Casos manuales (`08_tests/handwritten_cases/`) | Control de la circularidad | No |
| Motor auditor (`03_auditor/`) | Instrumento de medición determinista | No |

## Productos

**Skills** (`05_skills/`): copie la carpeta de la Skill a `~/.claude/skills/` (o al directorio de Skills de su agente).
- `audit-ml-leakage` → audita y repara scripts existentes (`python scripts/run_ast_audit.py archivo.py --fix`).
- `create-leakage-free-pipeline` → plantilla y reglas para crear pipelines sin fuga, con autoverificación.

**Plataforma web** (`06_web_app/`): pegue o suba un `.py`, vea las líneas con fuga, descargue el código corregido y
consulte los resultados del experimento. **Solo lee el código (nunca lo ejecuta)**, rechaza entradas de más de
200.000 caracteres y escapa el HTML. Use la web si no trabaja con un agente de programación; use las Skills si sí.

## Limitaciones que conviene decir antes de que las pregunten
- El AST sigue variables por nombre: no ve alias de importación (`import … as SS`), `exec`, ni fugas que dependen de
  valores (duplicados). El verificador en ejecución cubre parte de eso.
- «0 fugas detectadas» significa *sin evidencia de las reglas R-Split, R-CV y R-Test*, no *sin ningún tipo de fuga*.
- El banco de 192 scripts lo generó el mismo autor que escribió el auditor (evaluación circular); por eso existe la
  evaluación independiente con 13 scripts manuales (recall híbrido 9/9, IC95 % [0,66; 1,00]).
- Los **pipelines escritos por LLM reales no se han generado todavía** (Capstone II). Detalle y desviaciones:
  `07_report/decision_log.md`.

## Reproducibilidad
Python 3.13, versiones fijas en `requirements.txt`, semilla única 42, sin internet salvo la primera descarga de
`california_housing` (scikit-learn la guarda en caché). `run_benchmark.py` ordena todas sus tablas y escribe
`results/run_manifest.json` (versiones y semilla) y `results/benchmark_log.txt`, de modo que dos ejecuciones producen
archivos idénticos.
