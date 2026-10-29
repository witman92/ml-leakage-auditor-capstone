# Checklist contra la rúbrica v2.0 (Capstone I · Entrega Final)

Fuente: `rubrica_capstone-I-v2.0.pdf`. Los niveles son **una lectura de la rúbrica, no la nota oficial**.
Escala: Excelente 100–86 (6,0–7,0) · Bien 85–73 (5,0–5,9) · Regular 72–60 (4,0–4,9) · Deficiente 59–0.
Estado al 29-oct-2026, después de la refactorización V1 (detalle en `decision_log.md`, sección E).

## Requisitos de cumplimiento (condición para que evalúen)

| Req. | Estado | Evidencia / qué falta |
|---|---|---|
| R1 Integridad: referencias verificables, sin datos fabricados | ✅ con una tarea | Todas las cifras salen de `04_experiments/results/` y se regeneran con `python run_all.py`. **Verificar las 7 referencias clásicas añadidas en V1** (bloque final de `latex/references.bib`). |
| R2 Formato: plantilla UDLA, APA 7, **15–20 páginas** de cuerpo | ✅ | `latex/capstone_report_ef.pdf` con `udla-capstone.cls`; medir el cuerpo (cap. 1 a 7) tras cada cambio. |
| R3 Retroalimentación de E1 incorporada | ✅ (revisar) | Matriz de trazabilidad R1–R9 (Tabla 1.4). Comparar con el correo original de E1. |
| R4 Profesor Guía en portada | ✅ | Francisco Pérez; confirmar que figura en la Ficha de Inscripción. |

## Los 7 criterios

| # | Criterio (peso EF) | Evidencia en el informe | Nivel estimado |
|---|---|---|---|
| 1 | Formulación (20 %) | Problema, pregunta, OE1–OE4 verificables con evidencia (Tabla 1.2), criterios con umbral (Tabla 1.3), alcance/supuestos/restricciones, fallos ruidosos vs. silencioso (Tabla 1.1) | Excelente |
| 2 | Fundamentación (20 %) | Síntesis comparativa de 4 enfoques con fortalezas y limitaciones (Tabla 2.1); alternativas de métrica y de verdad de referencia; brecha; 6 decisiones de diseño con alternativa descartada y respaldo (Tabla 2.2) | Excelente |
| 3 | Diseño y datos (25 %) | Protocolo de 5 fases ligado a CRISP-DM, OE, script y artefacto (Tabla 3.1); EDA con calidad, representatividad y limitaciones; arquitectura generador/auditor (Tabla 3.3); hoja de ruta con hitos y desviaciones | Excelente |
| 4 | Rigor y validación (10 %) | Protocolo sin fuga (mismos pliegues, unidad = par); tres líneas base justificadas; IC95 % Clopper–Pearson y bootstrap; Holm y Bonferroni calculados; control positivo | Excelente |
| 5 | Interpretación (10 %) | Cap. 5: qué indican los resultados, factibilidad, limitaciones, riesgos con evidencia y ajustes | Excelente si el autor lo defiende con sus palabras |
| 6 | Aplicabilidad (5 %) | Usuarios y decisión; Skills, CLI y web en Streamlit con diseño en capas; requisitos; matriz de riesgos con medida y verificación (Tabla 6.1) | Excelente/Bien (falta piloto con usuarios: Capstone II) |
| 7 | Comunicación (10 %) | Figuras y tablas numeradas y citadas; cifras idénticas a los CSV; APA 7 con biblatex-apa | Excelente si se mantiene la consistencia de cifras |

## Riesgos que la rúbrica v2 penaliza explícitamente

1. **«El estudiante debe poder explicar y defender cualquier pasaje, las cifras y los términos»** (criterios 5 y 7). Ensayar en voz alta:
   qué hace cada regla del auditor, por qué ΔOpt = 0 en árboles con escalado, por qué el 100 % del banco no prueba nada, qué es un
   intervalo de Clopper–Pearson y qué hace el ajuste de Holm.
2. **Afirmaciones exploratorias.** El efecto de la MLP en regresión (p = 0,037; IC95 % que excluye el 0) **no** sobrevive a Holm
   (p = 0,297): no presentarlo como confirmado.
3. **Consistencia de cifras.** Si se vuelve a ejecutar el benchmark, regenerar el PDF y comparar con los CSV.
4. **Declaración de uso de IA** (cap. 6): está redactada como borrador; ajustarla a la política del programa.

## Pendientes del autor antes de entregar
- [x] Referencias nuevas verificadas contra fuentes primarias (Crossref/editorial, 29-oct-2026): Ambroise y McLachlan (2002), Varma y Simon (2006), Pedregosa et al. (2011), Clopper y Pearson (1934), Holm (1979), Hastie et al. (2009), Efron y Tibshirani (1993).
- [x] Declaración de uso de IA ajustada: ideas, diseño y borrador del autor; IA como apoyo de implementación y edición.
- [x] Matriz R1–R9 contrastada con `informe para correccion.md` (falta el correo original de E1, si difiere).
- [ ] Ensayar la defensa de los cap. 5 y 6 (se cubre al preparar la presentación).
- [x] Word regenerado desde el LaTeX (`docx/capstone_report_ef.docx`); el Word antiguo se eliminó.
- [ ] Revisar en conjunto y hacer commit y push al repositorio público.
