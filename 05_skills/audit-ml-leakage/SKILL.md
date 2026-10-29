---
name: audit-ml-leakage
description: Audita y repara fuga de datos (data leakage) en scripts de Machine Learning con scikit-learn. Úsala cuando el usuario pida revisar, validar o corregir un pipeline de clasificación o regresión, o cuando un modelo muestre métricas sospechosamente altas. Detecta escaladores, imputadores, selectores de variables y PCA ajustados con todas las filas antes de separar entrenamiento y prueba o fuera de un Pipeline dentro de validación cruzada.
---

# audit-ml-leakage — Auditoría y reparación de fuga de datos

## Qué problema resuelve
Un transformador (`StandardScaler`, `SimpleImputer`, `KNNImputer`, `SelectKBest`, `PCA`…) aprende
parámetros con `.fit()`. Si aprende con **todas** las filas, ya vio las filas de prueba y la métrica
queda inflada. El código **no falla ni avisa**: por eso hay que auditarlo.

## Cuándo usarla
- El usuario entrega un `.py` (o un cuaderno exportado) con un pipeline de scikit-learn y pide revisarlo.
- Un modelo da AUC/R² muy altos con pocos datos o con muchas variables.
- Antes de dar por bueno un pipeline generado por un LLM.

## Procedimiento (siga los pasos en orden)

1. **Auditar** con la herramienta incluida (no adivine: ejecútela):
   ```bash
   python scripts/run_ast_audit.py ARCHIVO.py --json
   ```
   Devuelve `estado` (`FUGA_DETECTADA` | `LIMPIO`), y por cada hallazgo: línea, tipo
   (`ESCALADO`, `IMPUTACION`, `SELECCION`, `REDUCCION`, `SOBREMUESTREO`), regla y confianza.

2. **Si está `LIMPIO`**: informe al usuario "0 fugas detectadas" **y la advertencia de alcance**
   (ver Limitaciones). No afirme que el pipeline "no tiene fuga" en términos absolutos.

3. **Si hay `FUGA_DETECTADA`**: intente la reparación automática:
   ```bash
   python scripts/run_ast_audit.py ARCHIVO.py --fix
   ```
   Crea `ARCHIVO_sin_fuga.py`. Cubre dos patrones: (a) transformadores + `cross_val_score`/`cross_validate`
   (los encierra en un `Pipeline`) y (b) transformadores + `train_test_split` (mueve el ajuste después de
   partir).

4. **Si la reparación automática no aplica** (el programa lo dice: *requiere revisión manual*),
   reescriba usted el código con estas reglas y **no cambie** el modelo, sus hiperparámetros ni las semillas:
   - Cada transformador va dentro de `sklearn.pipeline.Pipeline` (o `ColumnTransformer` dentro de uno), junto con el estimador.
   - Con `train_test_split`: parta primero; `fit_transform` solo sobre el train, `transform` sobre el test.
   - Con validación cruzada o `GridSearchCV`: pase el **Pipeline completo** y la matriz **original**.

5. **Verificar**: vuelva a ejecutar el paso 1 sobre el archivo corregido. El resultado esperado es
   `0 FUGAS DETECTADAS`. Si sigue habiendo hallazgos, repita el paso 4.

6. **Informe al usuario** (breve): qué fugas había (tipo y línea), qué cambió, y el resultado de la
   re-auditoría. Si el usuario quiere medir cuánto se inflaba la métrica, ejecute ambas versiones con los
   **mismos pliegues** y reporte la diferencia (sesgo de optimismo ΔOpt = métrica con fuga − métrica sin fuga).

## Limitaciones (decláralas siempre)
- Análisis **estático**: sigue variables por nombre y no ejecuta el código. Puede no ver importaciones con
  alias (`import ... as SS`), código generado dinámicamente (`exec`) ni librerías fuera de scikit-learn.
- No detecta duplicados entre entrenamiento y prueba ni fugas temporales o de dominio.
- La confianza `media` significa "sospecha": revise a mano esa línea.
- Una auditoría `LIMPIO` significa *sin evidencia de las reglas R-Split, R-CV y R-Test*, no *sin ningún tipo de fuga*.

## Reglas formales que aplica la herramienta
- **R-Split**: `.fit()`/`.fit_transform()` antes de `train_test_split`.
- **R-CV**: transformador ajustado fuera de un `Pipeline` cuyo resultado entra a `cross_val_score`/`cross_validate`/`GridSearchCV`.
- **R-Test**: transformador ajustado con una variable que parece de prueba (`X_test`, `X_te`, `X_val`…).
