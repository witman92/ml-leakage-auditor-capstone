# Prompt estándar (Conjunto A: pipelines escritos por LLM)

Lo lee `generate_with_llm.py` (toma el primer bloque de código delimitado); la salida se audita con
`04_experiments/audit_llm_generated.py`.

Se entrega **exactamente el mismo texto** a todos los modelos generadores; solo cambian
`{TAREA}`, `{DATASET}`, `{CARGA}`, `{MODELO}` y `{METRICAS}`. Los modelos generadores NO son el auditor: el auditor es
determinista (AST + runtime) y nunca consulta a un LLM.

```
Escribe un script de Python completo y ejecutable que use scikit-learn para resolver una tarea de
{TAREA} con el dataset {DATASET}, cargado así:

    {CARGA}

Requisitos:
1. Maneja los valores faltantes (imputación).
2. Escala las variables numéricas.
3. Selecciona las 8 mejores variables.
4. Entrena un modelo {MODELO}.
5. Evalúa con validación cruzada de 5 pliegues (random_state=42) y muestra {METRICAS}.
6. Usa solo pandas, numpy y scikit-learn. Devuelve únicamente el código, sin explicaciones.
```

Por qué NO se menciona la fuga de datos en el prompt: se mide la tendencia *espontánea* del modelo.
(Una variante "con advertencia de fuga" queda planificada para Capstone II.)
