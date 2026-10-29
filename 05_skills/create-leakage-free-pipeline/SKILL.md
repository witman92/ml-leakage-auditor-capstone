---
name: create-leakage-free-pipeline
description: Genera pipelines de clasificación o regresión con scikit-learn que están libres de fuga de datos por construcción. Úsala cuando el usuario pida crear, escribir o entrenar un modelo predictivo tabular desde cero (preprocesamiento, selección de variables, validación cruzada o ajuste de hiperparámetros). Impone Pipeline/ColumnTransformer, prueba final apartada y autoverificación con el auditor AST antes de entregar el código.
---

# create-leakage-free-pipeline — Crear pipelines seguros por diseño

## Objetivo
Escribir código de ML tabular donde la fuga de datos **no pueda ocurrir**, y comprobarlo antes de entregarlo.

## Reglas obligatorias al generar código
1. **Prohibido el ajuste global.** Ningún objeto con `.fit()`/`.fit_transform()` se llama sobre la matriz completa
   `X` ni sobre `y` antes de particionar.
2. **Todo en un `Pipeline`.** Imputación, escalado, codificación (`OneHotEncoder`), reducción de dimensión
   (`PCA`), **selección de variables (`SelectKBest`, `SelectFromModel`)** y el estimador van juntos en
   `sklearn.pipeline.Pipeline`; use `ColumnTransformer` dentro del Pipeline cuando haya columnas numéricas y de texto.
3. **Prueba final aparte.** Haga `train_test_split` primero (con `stratify=y` en clasificación) y toque el
   conjunto de prueba **una sola vez**, al final.
4. **Validación y ajuste con el Pipeline completo.** `cross_val_score`, `cross_validate` y `GridSearchCV`
   reciben el **Pipeline** y la matriz **original** (nunca una matriz ya transformada).
5. **Reproducibilidad.** Semilla fija (`random_state=42`) en particiones, validación y modelos.
6. **Métricas coherentes con la tarea.** Clasificación: ROC-AUC (`roc_auc_ovr` si hay >2 clases) y F1;
   regresión: R² y RMSE.

## Procedimiento
1. Pregunte (o deduzca) la tarea, la columna objetivo y el tipo de columnas si no están claros.
2. Parta de la plantilla `templates/safe_sklearn_template.py`: cópiela y adapte solo los datos (sección 1) y,
   si hace falta, el modelo (sección 3). **No reordene los pasos.**
3. **Autoverificación (obligatoria antes de entregar):**
   ```bash
   python scripts/run_ast_audit.py ARCHIVO_GENERADO.py --json
   ```
   Debe devolver `"estado": "LIMPIO"`. Si no, corrija y repita; **no entregue** código con `FUGA_DETECTADA`.
4. Ejecute el script si el entorno lo permite y compruebe que corre sin errores.
5. Entregue el código con una nota breve: reglas aplicadas, resultado de la auditoría y su alcance.

## Qué declarar siempre
La auditoría estática cubre **fuga de preprocesamiento y selección** (R-Split, R-CV, R-Test). No detecta
duplicados entre filas, fugas temporales ni fugas de dominio (variables que ya contienen el resultado). Para
datos con orden temporal use validación temporal (`TimeSeriesSplit`) y avise que esta Skill no la verifica.
