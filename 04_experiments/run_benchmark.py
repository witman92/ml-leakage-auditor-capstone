# ==============================================================================
# run_benchmark.py · FASES 2, 3 y 4 del protocolo: auditoría, eficacia del auditor y ΔOpt
# ==============================================================================
"""
Ejecuta el banco de control completo (192 scripts) en un solo comando.

    python 04_experiments/run_benchmark.py                                     # los 192 scripts
    python 04_experiments/run_benchmark.py --datasets D1 D4 --familias RF LIN  # subconjunto rápido

Para cada script de 02_pipelines/controlled_pairs/:
  FASE 2  Auditoría estática (AST) + verificación en ejecución (gancho sobre .fit). Al ejecutar
          el script se obtienen además sus puntajes de validación cruzada.
  FASE 3  Cada veredicto se compara con la verdad conocida (el script es "leaky" o "clean" por
          construcción: oráculo programático) → matriz de confusión, precisión, recall, F1 y VPN.
  FASE 4  Sesgo de Optimismo  ΔOpt = M_leaky − M_clean  por par (mismos pliegues, mismo modelo).
          Cada script leaky se repara con el refactorizador y se comprueba que el reparado rinde
          exactamente igual que el clean.

Salidas (04_experiments/results/):
    scores_per_fold.csv · auditor_per_script.csv · auditor_confusion_matrix.csv
    auditor_per_leak_type.csv · delta_optimism_pairs.csv · fixer_validation.csv
    benchmark_log.txt · run_manifest.json
"""

# %% PASO 0 · Configuración del entorno e importaciones
# ------------------------------------------------------------------------------
# QUÉ HACE   : fija un hilo de cálculo por proceso y agrega al path las carpetas del auditor y
#              del generador de pares.
# POR QUÉ    : el benchmark usa varios procesos en paralelo; si cada uno abriera muchos hilos de
#              BLAS se pelearían por los núcleos. Las variables deben fijarse ANTES de importar numpy.
# ------------------------------------------------------------------------------
import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "03_auditor"))
sys.path.insert(0, str(RAIZ / "02_pipelines"))

import numpy as np                      # noqa: E402  (los imports van después de fijar los hilos)
import pandas as pd                     # noqa: E402
from ast_leakage_detector import analizar_archivo     # noqa: E402
from pipeline_fixer import corregir_archivo           # noqa: E402
from runtime_verifier import ejecutar_con_verificador  # noqa: E402
import generate_controlled_pairs as generador         # noqa: E402

SEMILLA = 42
CARPETA_PARES = RAIZ / "02_pipelines" / "controlled_pairs"
CARPETA_REPARADOS = RAIZ / "02_pipelines" / "repaired_pairs"
CARPETA_RESULTADOS = RAIZ / "04_experiments" / "results"
UMBRAL_DEGENERADO = -1.0   # R² por pliegue por debajo de esto = modelo que divergió


# %% PASO 1 · Utilidad: extraer los puntajes por pliegue
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: la variable global `resultados` de cada script del banco, que es la salida
#   de sklearn.model_selection.cross_validate ({"test_roc_auc": [5 valores], ...}).
# LÓGICA: se conservan solo las claves "test_*" y se quita el prefijo → {métrica: [5 floats]}.
# ------------------------------------------------------------------------------
def _extraer_scores(resultados):
    """De la salida de cross_validate saca {metrica: [valor por pliegue]}."""
    return {k.replace("test_", ""): [float(v) for v in vals]
            for k, vals in resultados.items() if k.startswith("test_")}


# %% PASO 2 · FASE 2: auditar, ejecutar y reparar UN script
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: un archivo D?_FAM_FUGA_version.py; el nombre codifica dataset, familia,
#   tipo de fuga y la verdad conocida ("leaky" = tiene fuga, "clean" = no).
# LÓGICA (se ejecuta en un proceso aparte):
#   2a. AST: lee el código sin ejecutarlo y aplica las reglas R-Split, R-CV y R-Test.
#   2b. Runtime: ejecuta el script con un gancho que cuenta las filas que recibe cada .fit();
#       si algún transformador ve tantas filas como el dataset completo, hay fuga.
#   2c. Si el script es leaky: el refactorizador lo repara, se guarda en repaired_pairs/ y se
#       vuelve a ejecutar para obtener sus puntajes (deben coincidir con los del clean).
# SALIDA: un dict serializable (fila de auditoría, puntajes por versión e info de reparación).
# ------------------------------------------------------------------------------
def procesar_script(ruta):
    """Procesa UN script (se ejecuta en un proceso aparte). Devuelve un dict serializable."""
    ruta = Path(ruta)
    dataset, familia, fuga, version = ruta.stem.split("_")
    inicio = time.time()

    estatico = analizar_archivo(ruta)                       # 2a: lee el código, no lo ejecuta
    dinamico = ejecutar_con_verificador(ruta)               # 2b: ejecución vigilada + puntajes de CV
    scores = _extraer_scores(dinamico["globales"]["resultados"])

    fila = {
        "script": ruta.name, "dataset": dataset, "familia": familia, "fuga": fuga, "version": version,
        "esperado_fuga": version == "leaky",
        "ast_fuga": estatico["estado"] == "FUGA_DETECTADA", "ast_tipos": "|".join(estatico["tipos"]),
        "runtime_fuga": dinamico["fuga"], "max_filas_ajuste": dinamico["max_filas_ajuste"],
        "n_total": dinamico["n_total"],
    }
    salida = {"fila": fila, "scores": {version: scores}, "reparado": None}

    if version == "leaky":                                  # 2c: reparar y re-ejecutar
        CARPETA_REPARADOS.mkdir(exist_ok=True)
        destino = CARPETA_REPARADOS / f"{dataset}_{familia}_{fuga}_reparado.py"
        reparacion = corregir_archivo(ruta, destino)
        info = {"script": ruta.name, "reparado": reparacion["reparado"],
                "auditoria_final": reparacion["auditoria_final"]["estado"], "mensaje": reparacion["mensaje"]}
        if reparacion["reparado"]:
            din2 = ejecutar_con_verificador(destino)
            salida["scores"]["reparado"] = _extraer_scores(din2["globales"]["resultados"])
            info["runtime_fuga_reparado"] = din2["fuga"]
        salida["reparado"] = info

    fila["segundos"] = round(time.time() - inicio, 1)
    return salida


# %% PASO 3 · FASE 3: métricas de detección contra la verdad conocida
# ------------------------------------------------------------------------------
# LÓGICA: positivo = "hay fuga". Se cuentan TP, TN, FP, FN y se derivan:
#   precisión = TP/(TP+FP)  → de lo que el auditor marca, cuánto es fuga real.
#   recall    = TP/(TP+FN)  → de las fugas reales, cuántas encuentra.
#   F1        = 2TP/(2TP+FP+FN)
#   VPN       = TN/(TN+FN)  → si el auditor dice "limpio", con qué probabilidad lo es.
# Si un denominador es 0 se devuelve NaN (no se inventa un valor).
# ------------------------------------------------------------------------------
def metricas_deteccion(esperado, predicho):
    """TP/TN/FP/FN y métricas. Positivo = 'hay fuga'."""
    esperado, predicho = np.asarray(esperado, bool), np.asarray(predicho, bool)
    tp = int((esperado & predicho).sum()); tn = int((~esperado & ~predicho).sum())
    fp = int((~esperado & predicho).sum()); fn = int((esperado & ~predicho).sum())
    div = lambda a, b: round(a / b, 4) if b else float("nan")      # evita dividir por cero
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "precision": precision, "recall": recall,
            "f1": f1, "VPN": div(tn, tn + fn)}


# %% PASO 4 · FASES 3 y 4: construir las tablas finales
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: la lista de dicts devuelta por procesar_script (uno por script).
# LÓGICA:
#   - Detector híbrido = AST o Runtime (declara fuga si cualquiera de los dos la ve).
#   - Por tipo de fuga: recall de cada detector y falsos positivos sobre los clean.
#   - ΔOpt por par: media de 5 pliegues del leaky − media del clean, misma métrica. Con neg_rmse
#     (mayor = mejor) la misma resta da el optimismo con el signo correcto.
#   - Par DEGENERADO: algún pliegue con R² < −1 (el modelo divergió, p. ej. MLPRegressor con
#     atípicos). Mide inestabilidad numérica, no fuga: se marca y se excluye de las pruebas
#     estadísticas (desviación C2 en decision_log.md).
#   - REPRODUCIBILIDAD: todas las tablas se ordenan por claves explícitas, así dos corridas dan
#     archivos idénticos sin importar en qué orden terminen los procesos paralelos.
# ------------------------------------------------------------------------------
def construir_tablas(salidas):
    """Convierte la lista de resultados por script en las tablas finales."""
    filas = [s["fila"] for s in salidas]
    df = pd.DataFrame(filas).sort_values("script").reset_index(drop=True)
    df["hibrido_fuga"] = df["ast_fuga"] | df["runtime_fuga"]

    confusion = []
    for nombre, col in (("AST (estático)", "ast_fuga"), ("Runtime (ejecución)", "runtime_fuga"),
                        ("Híbrido (AST o Runtime)", "hibrido_fuga")):
        confusion.append({"detector": nombre, **metricas_deteccion(df["esperado_fuga"], df[col])})
    por_tipo = []
    for fuga, g in df.groupby("fuga"):
        leaky = g[g["esperado_fuga"]]
        por_tipo.append({"fuga": fuga, "n_leaky": len(leaky),
                         "recall_ast": round(leaky["ast_fuga"].mean(), 4),
                         "recall_runtime": round(leaky["runtime_fuga"].mean(), 4),
                         "falsos_positivos_ast": int(g[~g["esperado_fuga"]]["ast_fuga"].sum()),
                         "falsos_positivos_runtime": int(g[~g["esperado_fuga"]]["runtime_fuga"].sum())})

    medias, por_fold, peor_r2 = {}, [], {}
    for s in salidas:
        f = s["fila"]
        for version, metricas in s["scores"].items():
            for metrica, valores in metricas.items():
                medias[(f["dataset"], f["familia"], f["fuga"], version, metrica)] = float(np.mean(valores))
                if metrica == "r2":
                    peor_r2[(f["dataset"], f["familia"], f["fuga"], version)] = float(np.min(valores))
                for i, v in enumerate(valores):
                    por_fold.append({"dataset": f["dataset"], "familia": f["familia"], "fuga": f["fuga"],
                                     "version": version, "metrica": metrica, "fold": i, "valor": v})
    pares = []
    for (d, fam, fuga, version, metrica), m_leaky in medias.items():
        if version != "leaky":
            continue
        m_clean = medias.get((d, fam, fuga, "clean", metrica))
        m_rep = medias.get((d, fam, fuga, "reparado", metrica))
        if m_clean is None:
            continue
        degenerado = any(peor_r2.get((d, fam, fuga, v), 0.0) < UMBRAL_DEGENERADO for v in ("leaky", "clean"))
        pares.append({
            "dataset": d, "tarea": generador.DATASETS[d], "familia": fam, "fuga": fuga, "metrica": metrica,
            "degenerado": degenerado,
            "M_leaky": m_leaky, "M_clean": m_clean,
            "delta_opt": m_leaky - m_clean,
            "diff_reparado_vs_clean": (m_rep - m_clean) if m_rep is not None else float("nan")})
    reparaciones = [s["reparado"] for s in salidas if s["reparado"]]

    pares = pd.DataFrame(pares).sort_values(["dataset", "familia", "fuga", "metrica"]).reset_index(drop=True)
    por_fold = (pd.DataFrame(por_fold)
                .sort_values(["dataset", "familia", "fuga", "version", "metrica", "fold"]).reset_index(drop=True))
    reparaciones = pd.DataFrame(reparaciones).sort_values("script").reset_index(drop=True)
    return df, pd.DataFrame(confusion), pd.DataFrame(por_tipo), pares, por_fold, reparaciones


# %% PASO 5 · Trazabilidad: manifiesto del entorno
# ------------------------------------------------------------------------------
# QUÉ HACE   : registra versiones de Python y librerías, semilla y número de scripts.
# POR QUÉ    : sin el entorno exacto no se puede reproducir una cifra. No incluye fecha ni hora
#              para que el archivo sea idéntico entre corridas equivalentes.
# ------------------------------------------------------------------------------
def manifiesto(n_scripts):
    import scipy
    import sklearn
    return {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "semilla": SEMILLA, "n_scripts": n_scripts, "sistema": platform.system()}


# %% PASO 6 · Interpretación analítica automática
# ------------------------------------------------------------------------------
# CÓMO SE INTERPRETA:
#   - Un 100 % en el banco es ESPERABLE: el generador y el auditor tienen el mismo autor
#     (evaluación circular). La generalización se mide en independent_evaluation.py.
#   - Reparados = n/n y diff_reparado_vs_clean = 0 indican que la reparación elimina la fuga sin
#     cambiar el modelo: el código reparado rinde exactamente como el limpio.
# ------------------------------------------------------------------------------
def interpretar(confusion, reparaciones, pares):
    hib = confusion.iloc[-1]
    dif = pares["diff_reparado_vs_clean"].abs().max() if len(pares) else float("nan")
    lineas = ["INTERPRETACIÓN",
              f"  · Híbrido en el banco: precisión {hib['precision']:.2f}, recall {hib['recall']:.2f}. "
              "Es una evaluación circular (mismo autor del generador y del auditor): confirma que las "
              "reglas cubren los patrones diseñados, no que generalicen.",
              f"  · Reparación: {int(reparaciones['reparado'].sum())}/{len(reparaciones)} scripts; diferencia "
              f"máxima reparado vs. limpio = {dif:.2e} (0 = la reparación no altera el modelo).",
              f"  · Pares degenerados (R² < {UMBRAL_DEGENERADO:g} en algún pliegue): "
              f"{int(pares[['dataset', 'familia', 'fuga']][pares['degenerado']].drop_duplicates().shape[0])}."]
    return "\n".join(lineas)


# %% PASO 7 · Ejecución de punta a punta
# ------------------------------------------------------------------------------
# Genera el banco si falta, procesa los scripts en paralelo, construye las tablas, las guarda y
# escribe el registro (benchmark_log.txt) y el manifiesto (run_manifest.json).
# ------------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=list(generador.DATASETS))
    ap.add_argument("--familias", nargs="*", default=list(generador.FAMILIAS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--salida", default=str(CARPETA_RESULTADOS))
    args = ap.parse_args()

    if not CARPETA_PARES.exists() or not any(CARPETA_PARES.glob("*.py")):
        print("Generando el banco de control...")
        generador.generar_todos()
    scripts = [p for p in sorted(CARPETA_PARES.glob("*.py"))
               if p.stem.split("_")[0] in args.datasets and p.stem.split("_")[1] in args.familias]
    print(f"{len(scripts)} scripts · {args.workers} procesos en paralelo")

    salidas, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futuros = {pool.submit(procesar_script, str(p)): p for p in scripts}
        for i, fut in enumerate(as_completed(futuros), 1):
            salidas.append(fut.result())            # si un script falla, el error se ve aquí
            if i % 10 == 0 or i == len(scripts):
                print(f"  {i}/{len(scripts)} listos ({time.time() - t0:.0f} s)")

    df, confusion, por_tipo, pares, por_fold, reparaciones = construir_tablas(salidas)
    out = Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)
    df.drop(columns="segundos").to_csv(out / "auditor_per_script.csv", index=False)
    confusion.to_csv(out / "auditor_confusion_matrix.csv", index=False)
    por_tipo.to_csv(out / "auditor_per_leak_type.csv", index=False)
    pares.to_csv(out / "delta_optimism_pairs.csv", index=False)
    por_fold.to_csv(out / "scores_per_fold.csv", index=False)
    reparaciones.to_csv(out / "fixer_validation.csv", index=False)

    registro = "\n".join([
        f"{len(scripts)} scripts auditados (banco de control)", "",
        "=== FASE 3 · Eficacia del auditor (verdad conocida por construcción) ===",
        confusion.to_string(index=False), "",
        "=== FASE 4 · Reparación automática ===",
        f"Reparados y auditados LIMPIO: {int(reparaciones['reparado'].sum())}/{len(reparaciones)}", "",
        interpretar(confusion, reparaciones, pares), ""])
    (out / "benchmark_log.txt").write_text(registro, encoding="utf-8")
    (out / "run_manifest.json").write_text(json.dumps(manifiesto(len(scripts)), indent=2) + "\n",
                                           encoding="utf-8")
    print("\n" + registro)
    print(f"Listo en {time.time() - t0:.0f} s. Resultados en {out}")


if __name__ == "__main__":
    main()
