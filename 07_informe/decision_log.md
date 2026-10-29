# Registro de decisiones y desviaciones (rúbrica v2: «plan actualizado, con las desviaciones explicadas»)

Fecha de referencia: 29-oct-2026. Cada fila dice **qué se cambió, por qué y qué efecto tuvo**.
Los resultados descartados se conservan en `04_experiments/results/corrida1_descartada/` para que se vea el antes y el después.

## A. Decisiones de diseño tomadas a partir del informe de reformulación

| # | Decisión | Motivo (observación del comité) |
|---|---|---|
| A1 | Se abandona el proyecto anterior (`tesis_consolidada_v3`: 19 etapas, jueces y anotadores simulados, datasets OpenML) y se reconstruye en una carpeta nueva y mínima. | R4/R7: el flujo era difícil de auditar y de explicar; la rúbrica v2 exige poder defender cada cifra. `tesis_consolidada_v3` queda intacta como archivo histórico. |
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
| C2 | Un `MLPRegressor` divergió en un pliegue (R² = −21 en california; promedio −3,86). | Regla fijada tras ver el problema: un par es **degenerado** si algún pliegue tiene R² < −1; se excluye de las pruebas y se lista en `pares_degenerados.csv`. | Es una exclusión *post hoc*: se declara como tal. |
| C3 | Con 50 columnas de ruido, la fuga por selección casi no infla el AUC en datasets de señal fuerte (breast_cancer, digits). | No se manipuló el diseño para «obtener» efecto. Se añadió `sensibilidad_ruido.py`, que mide el ΔOpt con 0, 50, 200 y 1000 columnas de ruido. | La conclusión pasa a ser condicional: el daño de la selección depende de la razón señal/ruido. |
| C4 | El auditor obtuvo 192/192 sobre el banco que generó el mismo equipo (**evaluación circular**). | Se añadió una evaluación independiente con 13 scripts escritos a mano (`08_pruebas/casos_manuales`). | Sus resultados (incluido el caso que el AST falla) se reportan como medida de generalización. |
| C5 | En la evaluación independiente, el caso `05` (`X_te`) escapó al AST porque la heurística buscaba la palabra «test». | La heurística compara por palabras (`te`, `val`, `eval`…). | Sigue siendo una heurística por nombre: se declara como limitación. |
