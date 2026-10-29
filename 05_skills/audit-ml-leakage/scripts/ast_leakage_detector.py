"""
ast_leakage_detector.py  ·  Auditor estático de fuga de datos (Objetivo Específico 2).

IDEA EN UNA FRASE
    Lee el código SIN ejecutarlo (con el módulo `ast` de Python) y avisa si algún
    transformador de scikit-learn (escalador, imputador, selector, PCA...) se ajusta
    con TODAS las filas antes de que los datos se separen en entrenamiento/prueba.

¿POR QUÉ ES UNA FUGA?
    Un transformador "aprende" parámetros con .fit() (media, varianza, vecinos, variables
    elegidas). Si aprende con todo el dataset, ya vio las filas de prueba: la evaluación
    queda artificialmente optimista.

LAS DOS REGLAS FORMALES
    R-Split : se llama .fit()/.fit_transform() ANTES de train_test_split(...).
    R-CV    : se llama .fit()/.fit_transform() fuera de un Pipeline y el resultado
              entra a cross_val_score / cross_validate / GridSearchCV
              (cada pliegue de la validación ya "vio" las demás filas).
    R-Test  : (regla extra) se ajusta un transformador con una variable cuyo nombre
              indica que son datos de prueba (X_test, X_val...).

LO QUE NO HACE (limitaciones honestas, se discuten en el informe)
    - No ejecuta el código: no puede ver fugas que dependen de valores (p. ej. duplicados).
    - Sigue variables por nombre, de forma simple; código muy dinámico (exec, getattr)
      puede escapar. Para eso existe el verificador en ejecución (runtime_verifier.py).

USO
    python 03_auditor/ast_leakage_detector.py mi_script.py          # informe legible
    python 03_auditor/ast_leakage_detector.py mi_script.py --json   # salida JSON
"""

import ast
import json
import sys
from pathlib import Path

# --- Catálogo: nombre de clase de scikit-learn -> tipo de fuga que produce ---------------
TRANSFORMADORES = {
    # Escalado y normalización
    "StandardScaler": "ESCALADO", "MinMaxScaler": "ESCALADO", "RobustScaler": "ESCALADO",
    "MaxAbsScaler": "ESCALADO", "PowerTransformer": "ESCALADO", "QuantileTransformer": "ESCALADO",
    # Imputación de faltantes
    "SimpleImputer": "IMPUTACION", "KNNImputer": "IMPUTACION", "IterativeImputer": "IMPUTACION",
    # Selección de variables (la más dañina: usa las etiquetas y)
    "SelectKBest": "SELECCION", "SelectPercentile": "SELECCION", "SelectFromModel": "SELECCION",
    "RFE": "SELECCION", "RFECV": "SELECCION", "VarianceThreshold": "SELECCION",
    # Reducción de dimensión
    "PCA": "REDUCCION", "TruncatedSVD": "REDUCCION",
    # Sobremuestreo (librería imbalanced-learn)
    "SMOTE": "SOBREMUESTREO", "RandomOverSampler": "SOBREMUESTREO",
}
METODOS_AJUSTE = {"fit", "fit_transform", "fit_resample"}
FUNCIONES_VALIDACION = {"cross_val_score", "cross_validate", "cross_val_predict",
                        "GridSearchCV", "RandomizedSearchCV"}
FUNCIONES_PARTICION = {"train_test_split"}
# Una variable "parece de prueba" si alguna de sus palabras (separadas por _) está en este conjunto:
# X_test, X_te, y_val, test_X... Es una heurística por nombre: se documenta como limitación.
PALABRAS_PRUEBA = {"test", "te", "val", "valid", "validation", "eval", "holdout"}


def _parece_prueba(nombre_variable):
    return bool(set(nombre_variable.lower().split("_")) & PALABRAS_PRUEBA)


def _nombre_llamada(nodo):
    """Nombre simple de una llamada: 'StandardScaler' para StandardScaler() o
    preprocessing.StandardScaler(); None si no es una llamada con nombre."""
    if isinstance(nodo, ast.Call):
        f = nodo.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return f.attr
    return None


def _nombres_en(nodo):
    """Todos los nombres de variable que aparecen dentro de una expresión."""
    return {n.id for n in ast.walk(nodo) if isinstance(n, ast.Name)}


def _objetivos(asignacion):
    """Nombres de variable que recibe una asignación (soporta a, b = ... )."""
    nombres = set()
    for destino in asignacion.targets:
        for n in ast.walk(destino):
            if isinstance(n, ast.Name):
                nombres.add(n.id)
    return nombres


class _Recolector(ast.NodeVisitor):
    """Recorre el árbol UNA vez y junta los hechos que necesitan las reglas."""

    def __init__(self):
        self.transformadores = {}   # variable -> clase ("scaler" -> "StandardScaler")
        self.ajustes = []           # llamadas .fit / .fit_transform sobre un transformador
        self.particiones = []       # (linea, nombres que usa) de cada train_test_split
        self.validaciones = []      # (linea, nombres que usa) de cada cross_val_*
        self.derivadas = {}         # variable -> clase del transformador que la "contamina"
        self.asignaciones = []      # todas las asignaciones, en orden, para el flujo de datos

    # -- asignaciones: aquí nacen los transformadores y se propaga la "contaminación" ----
    def visit_Assign(self, nodo):
        self.asignaciones.append(nodo)
        clase = _nombre_llamada(nodo.value)
        if clase in TRANSFORMADORES:          # scaler = StandardScaler()
            for nombre in _objetivos(nodo):
                self.transformadores[nombre] = clase
        self.generic_visit(nodo)

    # -- llamadas: ajustes, particiones y validaciones -----------------------------------
    def visit_Call(self, nodo):
        nombre = _nombre_llamada(nodo)
        usados = set().union(*[_nombres_en(a) for a in nodo.args]) if nodo.args else set()

        if nombre in FUNCIONES_PARTICION:
            self.particiones.append((nodo.lineno, usados))
        elif nombre in FUNCIONES_VALIDACION:
            usados |= set().union(*[_nombres_en(k.value) for k in nodo.keywords]) if nodo.keywords else set()
            self.validaciones.append((nodo.lineno, usados))
        elif isinstance(nodo.func, ast.Attribute) and nodo.func.attr in METODOS_AJUSTE:
            receptor = nodo.func.value
            clase = None
            if isinstance(receptor, ast.Name) and receptor.id in self.transformadores:
                clase = self.transformadores[receptor.id]       # scaler.fit_transform(X)
                variable = receptor.id
            elif _nombre_llamada(receptor) in TRANSFORMADORES:  # StandardScaler().fit_transform(X)
                clase = _nombre_llamada(receptor)
                variable = None
            if clase:
                self.ajustes.append({
                    "linea": nodo.lineno, "clase": clase, "metodo": nodo.func.attr,
                    "variable": variable, "argumentos": usados,
                })
        self.generic_visit(nodo)


def _propagar_flujo(rec):
    """
    Flujo de datos mínimo: si `X2 = scaler.fit_transform(X)` o `X2 = scaler.transform(X)`,
    entonces X2 está "contaminada"; y cualquier variable calculada a partir de X2 también.
    Devuelve {variable_contaminada: clase_del_transformador}.
    """
    contaminadas = {}
    variables_ajustadas = {a["variable"] for a in rec.ajustes if a["variable"]}
    for asig in rec.asignaciones:                        # en orden de aparición
        valor = asig.value
        # a) salida directa de un transformador que se ajustó globalmente
        if (isinstance(valor, ast.Call) and isinstance(valor.func, ast.Attribute)
                and valor.func.attr in METODOS_AJUSTE | {"transform"}):
            receptor = valor.func.value
            if isinstance(receptor, ast.Name) and receptor.id in variables_ajustadas:
                for n in _objetivos(asig):
                    contaminadas[n] = rec.transformadores[receptor.id]
                continue
            if _nombre_llamada(receptor) in TRANSFORMADORES and valor.func.attr in METODOS_AJUSTE:
                for n in _objetivos(asig):
                    contaminadas[n] = _nombre_llamada(receptor)
                continue
        # b) variable derivada de otra ya contaminada (X_train, X_test = split(X_con_fuga, y))
        previas = _nombres_en(valor) & set(contaminadas)
        if previas:
            origen = contaminadas[sorted(previas)[0]]
            for n in _objetivos(asig):
                contaminadas[n] = origen
    return contaminadas


def analizar_codigo(codigo):
    """
    Audita el texto de un script. Devuelve un diccionario:
        estado    : "FUGA_DETECTADA" | "LIMPIO" | "ERROR_SINTAXIS"
        tipos     : lista de tipos de fuga encontrados (ESCALADO, IMPUTACION, ...)
        hallazgos : lista con línea, clase, tipo, regla, mensaje, confianza y código
        resumen   : texto de una línea
    """
    try:
        arbol = ast.parse(codigo)
    except SyntaxError as error:
        return {"estado": "ERROR_SINTAXIS", "tipos": [], "hallazgos": [],
                "resumen": f"No se pudo leer el código: {error}"}

    rec = _Recolector()
    rec.visit(arbol)
    contaminadas = _propagar_flujo(rec)
    lineas = codigo.splitlines()
    hallazgos = []

    for aj in rec.ajustes:
        regla = confianza = mensaje = None
        antes_de_split = [l for l, _ in rec.particiones if l > aj["linea"]]
        hubo_split_antes = any(l < aj["linea"] for l, _ in rec.particiones)
        alimenta_cv = [l for l, usados in rec.validaciones if usados & set(contaminadas)]

        if antes_de_split:                                   # R-Split
            regla, confianza = "R-Split", "alta"
            mensaje = (f"{aj['clase']}.{aj['metodo']}() se ejecuta en la línea {aj['linea']}, "
                       f"ANTES de train_test_split (línea {min(antes_de_split)}): "
                       "el transformador aprende con las filas de prueba.")
        elif alimenta_cv:                                    # R-CV con flujo verificado
            regla, confianza = "R-CV", "alta"
            mensaje = (f"{aj['clase']}.{aj['metodo']}() ajusta con todas las filas fuera de un "
                       f"Pipeline y su resultado entra a la validación cruzada (línea {alimenta_cv[0]}): "
                       "cada pliegue de prueba ya fue visto por el transformador.")
        elif rec.validaciones and not hubo_split_antes:      # R-CV sin flujo claro
            regla, confianza = "R-CV", "media"
            mensaje = (f"{aj['clase']}.{aj['metodo']}() se llama fuera de un Pipeline y el script usa "
                       "validación cruzada; no se pudo seguir el flujo de datos, revisar a mano.")
        elif any(_parece_prueba(v) for v in aj["argumentos"]):                        # R-Test
            regla, confianza = "R-Test", "alta"
            mensaje = (f"{aj['clase']}.{aj['metodo']}() se ajusta con datos que parecen de prueba "
                       f"({sorted(aj['argumentos'])}).")

        if regla:
            hallazgos.append({
                "linea": aj["linea"], "clase": aj["clase"], "tipo": TRANSFORMADORES[aj["clase"]],
                "regla": regla, "confianza": confianza, "mensaje": mensaje,
                "codigo": lineas[aj["linea"] - 1].strip() if aj["linea"] <= len(lineas) else "",
            })

    tipos = sorted({h["tipo"] for h in hallazgos})
    estado = "FUGA_DETECTADA" if hallazgos else "LIMPIO"
    resumen = (f"{len(hallazgos)} fuga(s) detectada(s): {', '.join(tipos)}" if hallazgos
               else "0 FUGAS DETECTADAS")
    return {"estado": estado, "tipos": tipos, "hallazgos": hallazgos, "resumen": resumen}


def analizar_archivo(ruta):
    """Igual que analizar_codigo pero leyendo un archivo .py."""
    return analizar_codigo(Path(ruta).read_text(encoding="utf-8"))


def _imprimir(resultado, ruta):
    print(f"Archivo : {ruta}")
    print(f"Estado  : {resultado['estado']}  ({resultado['resumen']})")
    for h in resultado["hallazgos"]:
        print(f"  - línea {h['linea']:>3} [{h['tipo']}/{h['regla']}/{h['confianza']}] {h['mensaje']}")
        print(f"      > {h['codigo']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    informe = analizar_archivo(sys.argv[1])
    if "--json" in sys.argv:
        print(json.dumps(informe, ensure_ascii=False, indent=2))
    else:
        _imprimir(informe, sys.argv[1])
    sys.exit(1 if informe["estado"] == "FUGA_DETECTADA" else 0)  # código de salida útil en CI
