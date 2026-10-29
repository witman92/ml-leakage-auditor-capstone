"""
run_benchmark.py  ·  PASOS 2, 3 y 4 del experimento (todo en un solo comando).

    python 04_experiments/run_benchmark.py                 # los 192 scripts (≈ 10-20 min)
    python 04_experiments/run_benchmark.py --datasets D1 D4 --familias RF LIN   # subconjunto rápido

QUÉ HACE, EN ORDEN
    Para cada script del banco de control (02_pipelines/controlled_pairs/):
      PASO 2  Auditoría estática (AST) y verificación en ejecución (hook de .fit).
              Al ejecutarlo se obtienen además los puntajes de validación cruzada.
      PASO 3  Se compara cada veredicto con la verdad conocida (el script es "leaky" o "clean"
              por construcción) -> matriz de confusión, precisión, recall, F1 y VPN.
      PASO 4  Se calcula el Sesgo de Optimismo  ΔOpt = M_leaky − M_clean  por cada par
              (mismos pliegues, mismo modelo). Además, se repara cada script leaky con el
              refactorizador automático y se comprueba que el reparado da el mismo puntaje
              que el clean.

SALIDAS (en 04_experiments/results/)
    scores_por_fold.csv           puntaje de cada pliegue de cada script
    auditor_por_script.csv        veredicto del AST y del verificador, script por script
    auditor_confusion_matrix.csv  TP/TN/FP/FN + métricas por detector
    auditor_por_tipo.csv          recall por tipo de fuga
    delta_optimism_pairs.csv      ΔOpt de cada par (dataset × familia × fuga × métrica)
    fixer_validation.csv          resultado del refactorizador sobre los 96 scripts leaky
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Un hilo por proceso: así los 4-8 procesos paralelos no se pelean por los núcleos.
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
import generar_pares_controlados as generador         # noqa: E402

CARPETA_PARES = RAIZ / "02_pipelines" / "controlled_pairs"
CARPETA_REPARADOS = RAIZ / "02_pipelines" / "reparados"
CARPETA_RESULTADOS = RAIZ / "04_experiments" / "results"
UMBRAL_DEGENERADO = -1.0   # R² por pliegue por debajo de esto = modelo que divergió


def _extraer_scores(resultados):
    """De la salida de cross_validate saca {metrica: [valor por pliegue]}."""
    return {k.replace("test_", ""): [float(v) for v in vals]
            for k, vals in resultados.items() if k.startswith("test_")}


def procesar_script(ruta):
    """
    Trabajo de UN script (se ejecuta en un proceso aparte). Devuelve un dict serializable.
    """
    ruta = Path(ruta)
    dataset, familia, fuga, version = ruta.stem.split("_")
    inicio = time.time()

    # PASO 2a: auditoría estática (lee el código, no lo ejecuta).
    estatico = analizar_archivo(ruta)
    # PASO 2b: ejecución vigilada (cuenta filas en cada .fit) + puntajes de CV.
    dinamico = ejecutar_con_verificador(ruta)
    scores = _extraer_scores(dinamico["globales"]["resultados"])

    fila = {
        "script": ruta.name, "dataset": dataset, "familia": familia, "fuga": fuga, "version": version,
        "esperado_fuga": version == "leaky",
        "ast_fuga": estatico["estado"] == "FUGA_DETECTADA", "ast_tipos": "|".join(estatico["tipos"]),
        "runtime_fuga": dinamico["fuga"], "max_filas_ajuste": dinamico["max_filas_ajuste"],
        "n_total": dinamico["n_total"],
    }
    salida = {"fila": fila, "scores": {version: scores}, "reparado": None}

    # PASO 4b: reparar el script con fuga y comprobar que ya no tiene fuga ni optimismo.
    if version == "leaky":
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


def construir_tablas(salidas):
    """Convierte la lista de resultados por script en las tablas finales (PASOS 3 y 4)."""
    filas = [s["fila"] for s in salidas]
    df = pd.DataFrame(filas).sort_values("script").reset_index(drop=True)
    df["hibrido_fuga"] = df["ast_fuga"] | df["runtime_fuga"]    # el híbrido declara fuga si CUALQUIERA la ve

    # --- PASO 3: eficacia del auditor contra la verdad conocida ---------------------------
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

    # --- PASO 4: ΔOpt por par --------------------------------------------------------------
    medias, por_fold, peor_r2 = {}, [], {}
    for s in salidas:
        f = s["fila"]
        for version, metricas in s["scores"].items():
            for metrica, valores in metricas.items():
                medias[(f["dataset"], f["familia"], f["fuga"], version, metrica)] = float(np.mean(valores))
                if metrica == "r2":      # para detectar modelos que divergieron en algún pliegue
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
        # PAR DEGENERADO: algún pliegue de regresión con R² < UMBRAL_DEGENERADO (el modelo divergió, p. ej.
        # MLPRegressor con outliers). Su media no mide la fuga sino la inestabilidad: se marca y se excluye
        # de las pruebas estadísticas (decisión registrada en decision_log.md, DESVIACIÓN D2).
        degenerado = any(peor_r2.get((d, fam, fuga, v), 0.0) < UMBRAL_DEGENERADO for v in ("leaky", "clean"))
        pares.append({
            "dataset": d, "tarea": generador.DATASETS[d], "familia": fam, "fuga": fuga, "metrica": metrica,
            "degenerado": degenerado,
            "M_leaky": m_leaky, "M_clean": m_clean,
            # Con neg_rmse (mayor = mejor) la misma resta da el optimismo en RMSE con signo correcto.
            "delta_opt": m_leaky - m_clean,
            "diff_reparado_vs_clean": (m_rep - m_clean) if m_rep is not None else float("nan")})
    reparaciones = [s["reparado"] for s in salidas if s["reparado"]]
    return (df, pd.DataFrame(confusion), pd.DataFrame(por_tipo), pd.DataFrame(pares),
            pd.DataFrame(por_fold), pd.DataFrame(reparaciones))


def main():
    if hasattr(sys.stdout, "reconfigure"):          # tildes correctas en la consola de Windows
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
    df.to_csv(out / "auditor_por_script.csv", index=False)
    confusion.to_csv(out / "auditor_confusion_matrix.csv", index=False)
    por_tipo.to_csv(out / "auditor_por_tipo.csv", index=False)
    pares.to_csv(out / "delta_optimism_pairs.csv", index=False)
    por_fold.to_csv(out / "scores_por_fold.csv", index=False)
    reparaciones.to_csv(out / "fixer_validation.csv", index=False)

    print("\n=== PASO 3 · Eficacia del auditor (verdad conocida por construcción) ===")
    print(confusion.to_string(index=False))
    print("\n=== PASO 4 · Reparación automática ===")
    print(f"Reparados y auditados LIMPIO: {int(reparaciones['reparado'].sum())}/{len(reparaciones)}")
    print(f"\nListo en {time.time() - t0:.0f} s. Resultados en {out}")


if __name__ == "__main__":
    main()
