"""
pipeline_fixer.py  ·  Refactorizador automático (parte determinista de la Skill de reparación).

IDEA EN UNA FRASE
    Reescribe un script con fuga para que cada transformador se ajuste SOLO con los datos
    de entrenamiento, sin cambiar el algoritmo, los hiperparámetros ni las particiones.

DOS PATRONES QUE SABE ARREGLAR (los más comunes en código generado por LLM)

  Patrón CV  (regla R-CV)                      Patrón Split (regla R-Split)
  ----------------------------------------     -----------------------------------------
  X_esc = scaler.fit_transform(X)              X_esc = scaler.fit_transform(X)
  cross_val_score(modelo, X_esc, y, cv=cv)     X_tr, X_te, y_tr, y_te = train_test_split(X_esc, y)
          ||                                              ||
  Pipeline([("scaler", StandardScaler()),      X_tr, X_te, y_tr, y_te = train_test_split(X, y)
            ("modelo", modelo)])               X_tr = scaler.fit_transform(X_tr)   # solo train
  cross_val_score(pipeline, X, y, cv=cv)       X_te = scaler.transform(X_te)       # solo aplica

LO QUE NO ARREGLA (devuelve `requiere_revision_manual = True` con el motivo)
    Cualquier patrón distinto (funciones auxiliares, bucles, DataFrames con ColumnTransformer
    manual...). Ahí la Skill pide al agente LLM que haga el cambio y vuelva a auditar.

NOTA: el resultado se genera con ast.unparse, por lo que se pierden los comentarios y el
formato del script original. El comportamiento numérico se verifica en run_benchmark.py.
"""

import ast
from pathlib import Path

try:                                        # funciona como script suelto o como módulo
    from ast_leakage_detector import TRANSFORMADORES, FUNCIONES_VALIDACION, analizar_codigo
except ImportError:                         # pragma: no cover
    from .ast_leakage_detector import TRANSFORMADORES, FUNCIONES_VALIDACION, analizar_codigo


def _nombre(nodo):
    """Nombre simple de una llamada ('StandardScaler' en StandardScaler(...))."""
    f = getattr(nodo, "func", None)
    return f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)


def _es_asignacion_simple(nodo):
    return isinstance(nodo, ast.Assign) and len(nodo.targets) == 1 and isinstance(nodo.targets[0], ast.Name)


def _cadena_de_ajustes(cuerpo):
    """
    Busca en el cuerpo del módulo la secuencia
        t1 = Ctor(...)            (creación del transformador)
        Xa = t1.fit_transform(X, [y])   (ajuste global)
    Devuelve (creaciones, ajustes) donde
        creaciones = {variable: nodo_Assign}         ajustes = [(nodo_Assign, variable, entrada, salida, ¿usa_y?)]
    """
    creaciones, ajustes = {}, []
    for nodo in cuerpo:
        if not _es_asignacion_simple(nodo):
            continue
        destino, valor = nodo.targets[0].id, nodo.value
        if _nombre(valor) in TRANSFORMADORES:
            creaciones[destino] = nodo
        elif (isinstance(valor, ast.Call) and isinstance(valor.func, ast.Attribute)
              and valor.func.attr == "fit_transform"
              and isinstance(valor.func.value, ast.Name) and valor.func.value.id in creaciones
              and valor.args and isinstance(valor.args[0], ast.Name)):
            ajustes.append((nodo, valor.func.value.id, valor.args[0].id, destino, len(valor.args) > 1))
    return creaciones, ajustes


def _agregar_import(arbol, modulo, nombre):
    """Inserta `from modulo import nombre` si aún no existe."""
    for n in arbol.body:
        if isinstance(n, ast.ImportFrom) and n.module == modulo and any(a.name == nombre for a in n.names):
            return
    nuevo = ast.ImportFrom(module=modulo, names=[ast.alias(name=nombre)], level=0)
    posicion = 0                              # después de los imports existentes
    for i, n in enumerate(arbol.body):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            posicion = i + 1
    arbol.body.insert(posicion, nuevo)


def _corregir_cv(arbol, creaciones, ajustes):
    """Patrón CV: encierra los transformadores y el modelo en un Pipeline."""
    salida_final = ajustes[-1][3]
    # Llamada de validación cuyo 2.º argumento es la salida contaminada.
    for nodo in ast.walk(arbol):
        if (isinstance(nodo, ast.Call) and _nombre(nodo) in FUNCIONES_VALIDACION
                and len(nodo.args) >= 2 and isinstance(nodo.args[1], ast.Name)
                and nodo.args[1].id == salida_final):
            llamada = nodo
            break
    else:
        return False, "No se encontró la llamada de validación que usa la matriz transformada.", False

    pasos = []                                # [("scaler", StandardScaler(...)), ...] en orden
    for _, variable, _, _, _ in ajustes:
        constructor = creaciones[variable].value
        pasos.append(ast.Tuple(elts=[ast.Constant(variable), constructor], ctx=ast.Load()))
    pasos.append(ast.Tuple(elts=[ast.Constant("modelo"), llamada.args[0]], ctx=ast.Load()))
    pipeline = ast.Call(func=ast.Name("Pipeline", ast.Load()),
                        args=[ast.List(elts=pasos, ctx=ast.Load())], keywords=[])

    llamada.args[0] = pipeline                                   # el "modelo" pasa a ser el Pipeline
    llamada.args[1] = ast.Name(ajustes[0][2], ast.Load())        # y recibe la matriz ORIGINAL
    _agregar_import(arbol, "sklearn.pipeline", "Pipeline")
    return True, "Transformadores encapsulados en sklearn.pipeline.Pipeline (patrón CV).", True


def _corregir_split(arbol, creaciones, ajustes):
    """Patrón Split: mueve el ajuste de los transformadores a después de train_test_split."""
    salida_final = ajustes[-1][3]
    for posicion, nodo in enumerate(arbol.body):
        if (isinstance(nodo, ast.Assign) and not _es_asignacion_simple(nodo)
                and isinstance(nodo.value, ast.Call) and _nombre(nodo.value) == "train_test_split"
                and nodo.value.args and isinstance(nodo.value.args[0], ast.Name)
                and nodo.value.args[0].id == salida_final
                and isinstance(nodo.targets[0], ast.Tuple) and len(nodo.targets[0].elts) == 4):
            division, indice = nodo, posicion
            break
    else:
        return False, "No se reconoció `a, b, c, d = train_test_split(X_transformada, y, ...)`.", False

    nombres = [e.id for e in division.targets[0].elts]           # X_tr, X_te, y_tr, y_te
    x_tr, x_te, y_tr = nombres[0], nombres[1], nombres[2]
    division.value.args[0] = ast.Name(ajustes[0][2], ast.Load())  # partir la matriz ORIGINAL

    nuevas = []
    for _, variable, _, _, usa_y in ajustes:
        ajuste_args = [ast.Name(x_tr, ast.Load())] + ([ast.Name(y_tr, ast.Load())] if usa_y else [])
        nuevas.append(ast.Assign(                                 # X_tr = t.fit_transform(X_tr[, y_tr])
            targets=[ast.Name(x_tr, ast.Store())],
            value=ast.Call(ast.Attribute(ast.Name(variable, ast.Load()), "fit_transform", ast.Load()),
                           ajuste_args, [])))
        nuevas.append(ast.Assign(                                 # X_te = t.transform(X_te)
            targets=[ast.Name(x_te, ast.Store())],
            value=ast.Call(ast.Attribute(ast.Name(variable, ast.Load()), "transform", ast.Load()),
                           [ast.Name(x_te, ast.Load())], [])))
    arbol.body[indice + 1:indice + 1] = nuevas
    return True, "Ajuste de transformadores movido DESPUÉS de train_test_split (patrón Split).", False


def corregir_codigo(codigo):
    """
    Intenta reparar `codigo`. Devuelve dict:
        codigo_corregido, reparado (bool), requiere_revision_manual (bool), mensaje,
        auditoria_final (resultado del auditor sobre el código corregido)
    """
    antes = analizar_codigo(codigo)
    if antes["estado"] != "FUGA_DETECTADA":
        return {"codigo_corregido": codigo, "reparado": False, "requiere_revision_manual": False,
                "mensaje": "No hay fuga que corregir.", "auditoria_final": antes}

    arbol = ast.parse(codigo)
    creaciones, ajustes = _cadena_de_ajustes(arbol.body)
    if not ajustes:
        ok, mensaje, borrar_creaciones = False, "Los ajustes no están al nivel del módulo con el patrón `Xn = t.fit_transform(X)`.", False
    else:
        ok, mensaje, borrar_creaciones = _corregir_cv(arbol, creaciones, ajustes)
        if not ok:
            ok, mensaje, borrar_creaciones = _corregir_split(arbol, creaciones, ajustes)
        if not ok:
            mensaje = "No coincide con los patrones CV ni Split. " + mensaje

    if not ok:
        return {"codigo_corregido": codigo, "reparado": False, "requiere_revision_manual": True,
                "mensaje": mensaje, "auditoria_final": antes}

    # Se quitan las líneas originales que ya no hacen falta.
    a_borrar = {id(a[0]) for a in ajustes}                       # los `Xn = t.fit_transform(X)`
    if borrar_creaciones:                                        # CV: los `t = Ctor()` pasan al Pipeline
        a_borrar |= {id(creaciones[a[1]]) for a in ajustes}
    arbol.body = [n for n in arbol.body if id(n) not in a_borrar]
    ast.fix_missing_locations(arbol)
    corregido = ast.unparse(arbol) + "\n"
    despues = analizar_codigo(corregido)
    return {"codigo_corregido": corregido, "reparado": despues["estado"] == "LIMPIO",
            "requiere_revision_manual": despues["estado"] != "LIMPIO",
            "mensaje": mensaje, "auditoria_final": despues}


def corregir_archivo(ruta_entrada, ruta_salida=None):
    """Repara un archivo; si se da ruta_salida, escribe allí el código corregido."""
    resultado = corregir_codigo(Path(ruta_entrada).read_text(encoding="utf-8"))
    if ruta_salida and resultado["reparado"]:
        Path(ruta_salida).write_text(resultado["codigo_corregido"], encoding="utf-8")
    return resultado


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("Uso: python pipeline_fixer.py script_con_fuga.py [script_corregido.py]")
    r = corregir_archivo(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(r["mensaje"], "| reparado:", r["reparado"], "| auditoría final:", r["auditoria_final"]["resumen"])
    if not (len(sys.argv) > 2):
        print("-" * 70)
        print(r["codigo_corregido"])
