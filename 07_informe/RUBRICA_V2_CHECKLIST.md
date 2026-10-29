# Checklist contra la rúbrica v2.0 (Capstone I · Entrega Final)

Fuente: `rubrica_capstone-I-v2.0.pdf`. Los niveles son **mi lectura de la rúbrica, no la nota oficial**.
Escala: Excelente 100–86 (6,0–7,0) · Bien 85–73 (5,0–5,9) · Regular 72–60 (4,0–4,9) · Deficiente 59–0.

## Requisitos de cumplimiento (condición para que evalúen)

| Req. | Estado | Qué falta |
|---|---|---|
| R1 Integridad: referencias verificables, sin datos fabricados | ⚠️ Parcial | 6 referencias nuevas (marcadas con * en el informe: Cawley y Talbot, Demšar, etc.) **no se han verificado contra la fuente**. Verificarlas una por una. Todas las cifras salen de CSV reproducibles; no hay datos simulados. |
| R2 Formato: plantilla UDLA, APA 7, **15–20 páginas** de cuerpo | ✅ Hecho (revisar) | `latex/informe_EF.pdf` usa `udla-capstone.cls`; el cuerpo mide **17 páginas** (cap. 1 a 7). Falta confirmar con el profesor si agradecimientos/resumen/abstract cuentan como «extras». |
| R3 Retroalimentación de E1 incorporada | ⚠️ Pendiente de verificar | La tabla del informe usa las 9 observaciones del informe de reformulación. **Comparar con el correo original** (la rúbrica menciona estilo de redacción, uso de *captions*, debilidades de metodología). |
| R4 Profesor Guía en portada | ✅ | Francisco Pérez; confirmar que figura en la Ficha de Inscripción. |

## Los 7 criterios

| # | Criterio (peso EF) | Evidencia en el informe | Nivel estimado | Qué lo subiría |
|---|---|---|---|---|
| 1 | Formulación (20 %) | §1: problema, pregunta, OE1–OE4 verificables, alcance, supuestos, restricciones, criterios con umbral (Tabla 2) | Excelente/Bien alto | Mantener coherencia entre objetivos y resultados (ya están enlazados en las Tablas 2 y 4) |
| 2 | Fundamentación (20 %) | §2: tres enfoques, Tabla 3, brecha y 4 decisiones de diseño con cita | Bien | Es la sección más corta; agregar 2–3 fuentes sobre reparación automática de código y sobre selección de variables con fuga |
| 3 | Diseño y datos (25 %) | §3: CRISP-DM adaptado, EDA (Tabla 5, Figura 2), preparación trazable, limitaciones, plan y desviaciones | Bien/Excelente | Falta mostrar *calidad* más allá de faltantes (duplicados, outliers por dataset) |
| 4 | Rigor y validación (10 %) | §4.1–4.2: protocolo sin fuga, métricas, línea base; avance verificable (pipeline funcionando) | Excelente/Bien | Comparar contra una línea base externa (un *linter* o una regla simple) |
| 5 | Interpretación (10 %) | §5: qué indica, factibilidad, limitaciones, riesgos con ajustes | Bien/Excelente | Que el autor la reescriba **con sus palabras** y pueda defenderla; la rúbrica lo exige |
| 6 | Aplicabilidad (5 %) | §6: usuarios, decisión, implementación, requisitos, riesgos y medidas | Bien | Una prueba de uso con 2–3 personas (queda en Capstone II) |
| 7 | Comunicación (10 %) | Figuras y tablas numeradas; referencias APA | Depende de R3 y de la plantilla | Citar **cada** figura y tabla en el texto (ya está) y mantener cifras idénticas entre texto, tablas y figuras |

## Riesgos que la rúbrica v2 penaliza explícitamente

1. **«El estudiante debe poder explicar y defender cualquier pasaje, las cifras y los términos»** (criterios 5 y 7). Antes de entregar,
   explique en voz alta: qué hace cada regla del auditor, por qué ΔOpt = 0 en árboles con escalado, por qué el 100 % del banco no prueba
   nada, y cómo se calculó cada tabla. Si algo no puede explicarlo, simplifíquelo o quítelo.
2. **Afirmaciones sin respaldo.** Dos puntos del informe son *exploratorios* y así están escritos: el efecto de la red neuronal
   (p = 0,037 sin corregir) y las comparaciones entre familias. No los presente como confirmados.
3. **Consistencia de cifras.** Si vuelve a ejecutar el benchmark, las cifras del informe pueden cambiar en el cuarto decimal: regenere
   las tablas desde `results/` y revise el informe.
4. **Declaración de uso de IA** (§6): complétela según la política del programa.

## Pendientes antes de entregar (en orden)
1. Verificar las referencias marcadas con *.
2. Comparar la tabla de retroalimentación con el correo de E1.
3. ~~Pasar a la plantilla UDLA y medir las páginas~~ (hecho: 17 páginas). Completar en el texto el marcador `[COMPLETAR]` de la declaración de uso de IA (el enlace de GitHub ya está).
4. Reescribir §5 y §6 con sus palabras; completar la declaración de IA.
5. ~~Subir el repositorio a GitHub~~ (hecho): https://github.com/witman92/ml-leakage-auditor-capstone — el enlace ya está en el Anexo A del informe.
