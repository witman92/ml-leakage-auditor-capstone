# ==============================================================================
# app.py · Plataforma web "Auditor de fuga de datos" (Streamlit) — alternativa a las Skills
# ==============================================================================
"""
Interfaz gráfica para quien NO usa un agente de programación.

    streamlit run 06_web_app/app.py          (se abre en http://localhost:8501)

Flujo: el usuario pega o sube su script → el auditor AST marca las líneas con fuga en rojo →
reparación automática con diff y descarga → consulta de ΔOpt y de los resultados del experimento.

SEGURIDAD: la app solo LEE el código (análisis AST); NUNCA lo ejecuta, y limita el tamaño de la
entrada (app_logic.MAX_CARACTERES). La verificación en ejecución existe solo en la línea de
comandos y en el benchmark, sobre código del propio proyecto.

Pestañas: 1. Auditar y reparar · 2. Cuánto infla la fuga (ΔOpt) · 3. Resultados · 4. Qué es la fuga
"""

# %% PASO 0 · Configuración de la página
# ------------------------------------------------------------------------------
# ARQUITECTURA: app.py (solo interfaz) → app_logic.py (lógica pura, probada con pytest) →
#   03_auditor/ (el mismo motor que usan las Skills).
# ------------------------------------------------------------------------------
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app_logic as logica  # noqa: E402

AVISO_ALCANCE = ("«0 fugas detectadas» significa *sin evidencia* de las reglas R-Split, R-CV y R-Test; "
                 "**no** significa «sin fuga». El análisis es estático: no ve alias de importación, código "
                 "dinámico ni fugas temporales o por duplicados (ver pestaña 4).")

st.set_page_config(page_title="Auditor de fuga de datos", page_icon="🔍", layout="wide")
st.title("🔍 Auditor de fuga de datos en pipelines de Machine Learning")
st.caption("Capstone I · MDS UDLA · Detecta y repara fugas de datos en scripts de scikit-learn "
           "(escalado, imputación, selección de variables, PCA). El código se analiza, nunca se ejecuta.")

pestana_auditar, pestana_delta, pestana_resultados, pestana_guia = st.tabs(
    ["1. Auditar y reparar", "2. Cuánto infla la fuga", "3. Resultados del experimento", "4. ¿Qué es la fuga?"])

# %% PASO 1 · Pestaña "Auditar y reparar"
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: texto pegado, archivo .py subido o un caso de ejemplo de
#   08_tests/handwritten_cases. LÓGICA: validar tamaño → auditar → resaltar → reparar (opcional).
# ------------------------------------------------------------------------------
with pestana_auditar:
    ejemplos = logica.lista_de_ejemplos()
    col_entrada, col_salida = st.columns(2)

    with col_entrada:
        st.subheader("Tu código")
        origen = st.radio("Origen", ["Pegar código", "Subir archivo .py", "Usar un ejemplo"], horizontal=True)
        codigo = ""
        if origen == "Pegar código":
            codigo = st.text_area("Pega aquí tu script", height=320, key="codigo_pegado")
        elif origen == "Subir archivo .py":
            archivo = st.file_uploader("Archivo .py", type=["py"])
            codigo = archivo.getvalue().decode("utf-8", errors="replace") if archivo else ""
        else:
            nombre = st.selectbox("Ejemplo", list(ejemplos)) if ejemplos else None
            codigo = ejemplos[nombre].read_text(encoding="utf-8") if nombre else ""
            st.code(codigo, language="python")

    with col_salida:
        st.subheader("Resultado")
        valido, mensaje = logica.validar_entrada(codigo)
        if not valido:
            (st.info if not codigo.strip() else st.error)(mensaje)
        else:
            informe = logica.auditar(codigo)
            if informe["estado"] == "ERROR_SINTAXIS":
                st.error(informe["resumen"])
            elif informe["estado"] == "LIMPIO":
                st.success("✅ 0 fugas detectadas")
                st.warning(AVISO_ALCANCE)
            else:
                st.error(f"⚠️ {informe['resumen']}")
                for h in informe["hallazgos"]:
                    st.markdown(f"**Línea {h['linea']}** · `{h['tipo']}` · regla `{h['regla']}` · "
                                f"confianza **{h['confianza']}**")
                    st.write(h["mensaje"])
                st.markdown(logica.html_codigo_resaltado(codigo, {h["linea"] for h in informe["hallazgos"]}),
                            unsafe_allow_html=True)

                if st.button("🛠️ Reparar automáticamente"):
                    reparacion = logica.reparar(codigo)
                    if reparacion["reparado"]:
                        st.success(f"Reparado. {reparacion['mensaje']} "
                                   f"Re-auditoría: {reparacion['auditoria_final']['resumen']}")
                        st.code(logica.diff_unificado(codigo, reparacion["codigo_corregido"]), language="diff")
                        st.download_button("⬇️ Descargar código corregido", reparacion["codigo_corregido"],
                                           file_name="pipeline_sin_fuga.py", mime="text/x-python")
                    else:
                        st.warning("No se pudo reparar automáticamente: " + reparacion["mensaje"] +
                                   " Encierra los transformadores en un `sklearn.pipeline.Pipeline` "
                                   "(guía en la pestaña 4).")

# %% PASO 2 · Pestaña "Cuánto infla la fuga" (ΔOpt)
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: delta_optimism_pairs.csv. El usuario elige dataset, familia, fuga y métrica
#   y ve M_leaky, M_clean y ΔOpt del par controlado correspondiente.
# ------------------------------------------------------------------------------
with pestana_delta:
    resultados = logica.cargar_resultados()
    st.subheader("Sesgo de Optimismo ΔOpt = métrica con fuga − métrica sin fuga")
    st.write("Mismo dataset, mismos pliegues, mismo modelo: lo único que cambia es si el preprocesamiento "
             "se ajustó con todas las filas (con fuga) o dentro de un Pipeline (sin fuga).")
    if resultados is None:
        st.warning("Todavía no hay resultados. Ejecuta `python 04_experiments/run_benchmark.py` "
                   "y vuelve a abrir la app.")
    else:
        pares = resultados["pares"]
        c1, c2, c3, c4 = st.columns(4)
        dataset = c1.selectbox("Dataset", sorted(pares["dataset"].unique()))
        familia = c2.selectbox("Familia", ["RF", "GB", "LIN", "MLP"])
        fuga = c3.selectbox("Tipo de fuga", ["ESCALADO", "IMPUTACION", "SELECCION", "COMBINADA"])
        metricas = sorted(pares[pares["dataset"] == dataset]["metrica"].unique())
        metrica = c4.selectbox("Métrica", metricas)
        fila = logica.par_controlado(pares, dataset, familia, fuga, metrica)
        if fila is None:
            st.info("No hay datos para esa combinación.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Con fuga", f"{fila['M_leaky']:.4f}")
            m2.metric("Sin fuga (real)", f"{fila['M_clean']:.4f}")
            m3.metric("ΔOpt (optimismo)", f"{fila['delta_opt']:+.4f}")
            if abs(fila["delta_opt"]) < 1e-9:
                st.caption("ΔOpt = 0: esta fuga no cambia el resultado de esta familia (p. ej., el escalado no "
                           "altera un árbol de decisión).")
            if bool(fila.get("degenerado", False)):
                st.caption("Par degenerado: el modelo divergió en algún pliegue; se excluye de las pruebas.")
        st.markdown("**Mapa de calor del experimento completo**")
        ruta_fig = logica.RESULTADOS / "fig_delta_opt_heatmap.png"
        if ruta_fig.exists():
            st.image(str(ruta_fig))

# %% PASO 3 · Pestaña "Resultados del experimento"
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: matriz de confusión, recall por tipo, pruebas por familia (con Holm e IC95 %),
#   Friedman y EDA. Se recuerda que el 100 % del banco es una evaluación circular.
# ------------------------------------------------------------------------------
with pestana_resultados:
    resultados = logica.cargar_resultados()
    if resultados is None:
        st.warning("Sin resultados todavía. Ejecuta el benchmark (ver README).")
    else:
        st.subheader("Fase 3 · ¿Qué tan bien detecta el auditor? (verdad conocida por construcción)")
        st.dataframe(resultados["confusion"], use_container_width=True)
        st.caption("Positivo = «hay fuga». VPN = probabilidad de que un script declarado limpio lo esté de verdad. "
                   "Este banco lo generó el mismo autor que escribió el auditor (evaluación circular); la prueba "
                   "independiente está en 08_tests/handwritten_cases.")
        if "por_tipo" in resultados:
            st.dataframe(resultados["por_tipo"], use_container_width=True)
        st.subheader("Fase 4 · ¿Qué familias sufren más?")
        if "tests" in resultados:
            st.dataframe(resultados["tests"], use_container_width=True)
            st.caption("p_wilcoxon_holm: valor p ajustado por multiplicidad (8 pruebas). ic95_inf/ic95_sup: "
                       "intervalo bootstrap de la media de ΔOpt. Un efecto solo significativo sin corregir es exploratorio.")
        if "friedman" in resultados:
            st.dataframe(resultados["friedman"], use_container_width=True)
        ruta_barras = logica.RESULTADOS / "fig_delta_opt_bars.png"
        if ruta_barras.exists():
            st.image(str(ruta_barras))
        if "eda" in resultados:
            st.subheader("Datasets (análisis exploratorio)")
            st.dataframe(resultados["eda"], use_container_width=True)

# %% PASO 4 · Pestaña "¿Qué es la fuga?" (guía didáctica y limitaciones)
# ------------------------------------------------------------------------------
with pestana_guia:
    st.subheader("¿Qué es la fuga de datos?")
    st.write("Un transformador (escalador, imputador, selector…) *aprende* parámetros con `.fit()`. Si aprende con "
             "**todas** las filas, ya vio las filas que luego se usan para evaluar: la métrica sale inflada y el "
             "código **no da ningún error** (fallo silencioso).")
    izq, der = st.columns(2)
    izq.markdown("**❌ Con fuga**")
    izq.code("""scaler = StandardScaler()
X_esc = scaler.fit_transform(X)      # ve TODAS las filas
scores = cross_val_score(modelo, X_esc, y, cv=5)""", language="python")
    der.markdown("**✅ Sin fuga**")
    der.code("""pipe = Pipeline([("scaler", StandardScaler()),
                 ("modelo", modelo)])
# en cada pliegue, .fit() ve solo el entrenamiento
scores = cross_val_score(pipe, X, y, cv=5)""", language="python")
    st.markdown("""
**Reglas que aplica el auditor**
- `R-Split`: se ajusta un transformador **antes** de `train_test_split`.
- `R-CV`: se ajusta fuera de un `Pipeline` y el resultado entra a la validación cruzada.
- `R-Test`: se ajusta con una variable que parece de prueba (`X_test`, `X_te`, `X_val`).

**Limitaciones (importante)**
- Es análisis estático: no ejecuta el código. No ve alias de importación (`... as SS`), `exec`, ni fugas que
  dependen de valores (duplicados entre entrenamiento y prueba).
- «0 fugas detectadas» significa *sin evidencia de estas reglas*, no *sin ningún tipo de fuga*.
- Confianza `media` = sospecha: revisa esa línea a mano.
- Privacidad: el código se procesa en memoria y no se guarda.
""")
