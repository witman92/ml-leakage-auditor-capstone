# Registro de decisiones y desviaciones (rúbrica v2: «plan actualizado, con las desviaciones explicadas»)

Fecha de referencia: 29-oct-2026. Cada fila dice **qué se cambió, por qué y qué efecto tuvo**.
Los resultados descartados se conservan en `04_experiments/results/run1_discarded/` para que se vea el antes y el después.

## A. Decisiones de diseño tomadas a partir del informe de reformulación

| # | Decisión | Motivo (observación del comité) |
|---|---|---|
| A1 | Se abandona el proyecto anterior (`tesis_consolidada_v3`: 19 etapas, jueces y anotadores simulados, datasets OpenML) y se reconstruye en una carpeta nueva y mínima. | R4/R7: el flujo era difícil de auditar y de explicar; la rúbrica v2 exige poder defender cada cifra. `tesis_consolidada_v3` se eliminó el 29-oct-2026 para no generar ruido; sus referencias ya están en `references.bib`. |
| A2 | Datasets: 6 nativos de `scikit-learn` (3 clasificación, 3 regresión). Sin OpenML. | R5. |
| A3 | Sin anotadores humanos: la verdad conocida sale de construir pares *leaky/clean* (oráculo programático). | R5. |
| A4 | La métrica pasa a llamarse **Sesgo de Optimismo ΔOpt** = M(leaky) − M(clean), con los mismos pliegues. Se retira «brecha de generalización auditada». | R9: la brecha de generalización clásica mide sobreajuste, no fuga. |
| A5 | El producto final son dos Skills (`audit-ml-leakage`, `create-leakage-free-pipeline`) **más una plataforma web** como segunda alternativa para quien no use agentes. | R8 + petición del autor. |
| A6 | Los modelos LLM generadores (Conjunto A) quedan **preparados pero no ejecutados** (`02_pipelines/llm_generated/`). El auditor nunca usa un LLM. | R2. Requiere claves de pago; los resultados de Capstone I no dependen de ello. Se declara como alcance de Capstone II. |

## B. Desviaciones respecto del informe de reformulación (con su justificación)

| # | Desviación | Por qué | Efecto |
|---|---|---|---|
| B1 | `california_housing` se submuestrea a **3.000 filas** (de 20.640). | `KNNImputer` cuesta ~n²; con 20.640 filas el banco de 192 scripts deja de ser ejecutable en un portátil. | D5 es más pequeño que lo planificado. Se declara como limitación. |
| B2 | El oversampling (SMOTE) **no** se incluye como tipo de fuga; el cuarto tipo es **COMBINADA** (imputación + escalado + selección). | `imbalanced-learn` no es parte de `scikit-learn` (R5). El propio informe usa «L4 = combinada» en el paso 1. Había una inconsistencia interna (L4 = sobremuestreo en §5.1). | El catálogo del auditor reconoce SMOTE/RandomOverSampler, pero no se mide experimentalmente. |
| B3 | `LogisticRegression` con `max_iter=2000` y sin `penalty="l2"` explícito. | L2 es el valor por defecto; el límite de iteraciones evita avisos de no convergencia. | Ninguno en la definición del modelo. |
| B4 | Se agregan 50 columnas de ruido gaussiano a D1–D5 (D6 ya trae 45). | Pedido en el informe («columnas de ruido controladas»); sin ellas la fuga por selección no es visible. | Ver C2: con señal fuerte el efecto sigue siendo pequeño. |

## C. Hallazgos que obligaron a corregir el experimento (primera corrida → segunda corrida)

| # | Hallazgo en la corrida 1 | Corrección | Efecto |
|---|---|---|---|
| C1 | En fugas IMPUTACIÓN y SELECCIÓN, `LIN` y `MLP` se evaluaban **sin escalar**. Aparecían ΔOpt negativos grandes (p. ej. MLP/wine/SELECCIÓN = −0,155): inestabilidad numérica, no fuga. | Esas dos familias llevan un `StandardScaler` **interno** (dentro del modelo, igual en *leaky* y *clean*). | El escalado deja de ser una fuente de diferencia entre versiones. |
| C2 | Un `MLPRegressor` divergió en un pliegue (R² = −21 en california; promedio −3,86). | Regla fijada tras ver el problema: un par es **degenerado** si algún pliegue tiene R² < −1; se excluye de las pruebas y se lista en `degenerate_pairs.csv`. | Es una exclusión *post hoc*: se declara como tal. |
| C3 | Con 50 columnas de ruido, la fuga por selección casi no infla el AUC en datasets de señal fuerte (breast_cancer, digits). | No se manipuló el diseño para «obtener» efecto. Se añadió `noise_sensitivity.py`, que mide el ΔOpt con 0, 50, 200 y 1000 columnas de ruido. | La conclusión pasa a ser condicional: el daño de la selección depende de la razón señal/ruido. |
| C4 | El auditor obtuvo 192/192 sobre el banco que generó el mismo equipo (**evaluación circular**). | Se añadió una evaluación independiente con 13 scripts escritos a mano (`08_tests/handwritten_cases`). | Sus resultados (incluido el caso que el AST falla) se reportan como medida de generalización. |
| C5 | En la evaluación independiente, el caso `05` (`X_te`) escapó al AST porque la heurística buscaba la palabra «test». | La heurística compara por palabras (`te`, `val`, `eval`…). | Sigue siendo una heurística por nombre: se declara como limitación. |

## D. Correcciones de la versión V1 (29-oct-2026), contra el informe de corrección y la rúbrica v2.0

| # | Cambio | Por qué | Efecto |
|---|---|---|---|
| D1 | Se agrega `04_experiments/leakage_demo.py` y la Figura 1.2 (`fig_leakage_demo.png`). | R3: el informe de corrección pide un ejemplo paso a paso con gráfico. Se corrigió el código del informe: guarda el PNG (no usa `plt.show`), usa título en cadena cruda y no recorta el eje. | Con señal real ΔOpt = +0,0015; con señal nula (500 columnas de ruido, etiquetas permutadas) +0,048. Se informan ambos: el primero es casi invisible. |
| D2 | Se agrega `baseline_simple_rule.py` (regla trivial: `.fit` sin `Pipeline`). | Rúbrica criterio 4 (línea base). | Recall 0,75 en el banco y 0,89 en los 13 casos manuales, con 1 FP. **Primeras versiones de la regla daban recall 0** por contar `import Pipeline` y un docstring como uso real; se corrigió la regla (no el auditor) y se declara. |
| D3 | `eda_datasets.py` suma filas duplicadas y celdas atípicas (\|z\|>3). | Rúbrica criterio 3 (calidad más allá de faltantes). | 0 duplicados; atípicos entre 0 % y 1,24 %. |
| D4 | `cawley2010` y `demsar2006` verificadas en jmlr.org; se añaden `kapoor2024` (REFORMS) y `lai2023` (DS-1000), tomadas de la bibliografía del proyecto anterior (`tesis_consolidada_v3`, hoy eliminado). | Rúbrica R1 (referencias verificables) y criterio 2. | De v3 solo se reutilizaron referencias; ninguna cifra ni figura (sus datos eran simulados). |

## E. Refactorización V1 para la rúbrica v2.0 y el informe de corrección (29-oct-2026)

| # | Cambio | Por qué | Efecto |
|---|---|---|---|
| E1 | Nomenclatura MLOps en inglés para carpetas y archivos: `06_web`→`06_web_app`, `07_informe`→`07_report`, `08_pruebas`→`08_tests`, `casos_manuales`→`handwritten_cases` (casos renombrados `NN_leaky_*`/`NN_clean_*`, etiquetas en `expected_labels.json`), `reparados`→`repaired_pairs`, `ejecutar_todo.py`→`run_all.py`, `generar_pares_controlados.py`→`generate_controlled_pairs.py`, `demo_fuga.py`→`leakage_demo.py`, `linea_base_regla_simple.py`→`baseline_simple_rule.py`, `sensibilidad_ruido.py`→`noise_sensitivity.py`, `evaluacion_independiente.py`→`independent_evaluation.py`, `logica.py`→`app_logic.py`, `informe_EF.tex`→`capstone_report_ef.tex`; artefactos de `results/` en inglés (p. ej. `scores_per_fold.csv`, `tests_by_family.csv`). `requirements.txt` vuelve a la raíz. | Repositorio auditable por la industria; coincide con la estructura del informe de corrección (§9.1: `06_web_app/`). Los identificadores de Python y las etiquetas de datos (ESCALADO, IMPUTACION…) siguen en español. | Ninguno en las cifras. `run1_discarded/` conserva sus archivos con los nombres antiguos (es un archivo histórico). |
| E2 | Todos los scripts reescritos en bloques `# %% PASO n` con comentarios sobre origen de datos, lógica, métrica/gráfico e interpretación; cada análisis imprime un bloque INTERPRETACIÓN. | Informe de corrección R3/R7: poder explicar el código en la presentación. | Sin cambio de comportamiento: el banco regenerado solo cambia la línea del docstring y las cifras se reproducen. |
| E3 | `run_benchmark.py` ordena todas las tablas de salida, elimina la columna de tiempos y escribe `benchmark_log.txt` y `run_manifest.json` (versiones, semilla; sin marcas de tiempo). | Antes, el orden de filas dependía de qué proceso terminaba primero y dos corridas daban CSV distintos. | Dos corridas producen archivos idénticos. |
| E4 | `statistical_analysis.py` agrega IC95 % bootstrap de la media de ΔOpt (10.000 remuestreos, semilla 42) y el ajuste de Holm sobre las 8 pruebas; Friedman agrega Bonferroni. `independent_evaluation.py` y `baseline_simple_rule.py` agregan el IC95 % exacto de Clopper–Pearson del recall. | Rúbrica criterio 4: métricas sólidas y la multiplicidad, antes solo discutida en prosa, ahora calculada. | MLP en regresión: IC95 % [0,0027; 0,0346] pero p Holm = 0,297 → sigue siendo exploratorio. Recall híbrido independiente 9/9 con IC95 % [0,66; 1,00]. |
| E5 | `run_all.py` incluye la evaluación independiente (antes se ejecutaba aparte). | Trazabilidad: cada tabla del informe sale del comando único. | — |
| E6 | Plataforma web: límite de 200.000 caracteres (`app_logic.validar_entrada`), aviso visible de alcance cuando el resultado es «0 fugas»; la CLI imprime el mismo aviso y lo añade como clave `alcance` en `--json`. | Rúbrica criterio 6: medidas concretas y verificables contra la falsa seguridad y el código malicioso. | 3 pruebas nuevas (40 en total). |
| E7 | Informe: matriz de trazabilidad R1–R9 (sec. 1.7), fundamentación reescrita como síntesis comparativa de alternativas con tabla de decisiones (cap. 2), protocolo de cinco fases y arquitectura generador/auditor (cap. 3), tres líneas base justificadas (cap. 4), diseño de la plataforma Streamlit y matriz de riesgos con verificación (cap. 6), conclusiones que responden la pregunta. Se añaden 7 referencias clásicas (Ambroise y McLachlan, 2002; Varma y Simon, 2006; Hastie et al., 2009; Pedregosa et al., 2011; Holm, 1979; Clopper y Pearson, 1934; Efron y Tibshirani, 1993). | Criterios 1, 2, 3, 4, 6 y 7 de la rúbrica; informe de corrección R1–R9. | El autor debe verificar las 7 referencias nuevas y revisar la declaración de uso de IA (cap. 6), redactada como borrador. |
