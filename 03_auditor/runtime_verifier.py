# ==============================================================================
# runtime_verifier.py · Verificador en ejecución (segunda opinión del auditor)
# ==============================================================================
"""
Ejecuta el script de verdad y cuenta con cuántas FILAS se ajusta cada transformador. Si alguno se
ajusta con tantas filas como el dataset completo, vio el conjunto de prueba: hay fuga.

Uso:
    from runtime_verifier import ejecutar_con_verificador
    r = ejecutar_con_verificador("mi_script.py")
    r["fuga"], r["max_filas_ajuste"], r["n_total"], r["globales"]["resultados"]

ADVERTENCIA: ejecuta código. Úselo solo con scripts confiables o dentro de un entorno aislado
(la plataforma web 06_web_app NO usa este módulo por esa razón).
"""

# %% PASO 0 · Por qué hace falta si ya existe el auditor AST
# ------------------------------------------------------------------------------
# QUÉ HACE   : documenta el rol del verificador dentro del detector híbrido.
# POR QUÉ    : el AST lee el código pero no lo "entiende" del todo (alias, funciones auxiliares);
#              el verificador mide lo que PASÓ, no lo que el código parece decir. Son mecanismos
#              independientes: si coinciden, la conclusión es más sólida; si difieren, cada uno
#              cubre un punto ciego del otro (híbrido = AST o ejecución).
# REGLA DE DECISIÓN:  n_filas_ajuste >= n_total   =>   FUGA
#              (en 5 pliegues, un transformador bien encapsulado ve ≈ 80 % de las filas).
# LIMITACIÓN : no ve fugas que no usan todas las filas (p. ej. ajustar con el 70 % de
#              entrenamiento y luego hacer validación cruzada encima, o re-ajustar con X_test).
# ------------------------------------------------------------------------------
import runpy
import threading
import warnings

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import (RFE, SelectFromModel, SelectKBest, SelectPercentile,
                                       VarianceThreshold)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, PowerTransformer,
                                   QuantileTransformer, RobustScaler, StandardScaler)


# %% PASO 1 · Clases vigiladas
# ------------------------------------------------------------------------------
# QUÉ HACE   : lista los transformadores de scikit-learn cuyo .fit / .fit_transform se interceptan.
# NOTA       : coincide con el catálogo del AST salvo las clases de librerías externas (SMOTE).
# ------------------------------------------------------------------------------
CLASES_VIGILADAS = [
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer,
    SimpleImputer, KNNImputer,
    SelectKBest, SelectPercentile, SelectFromModel, RFE, VarianceThreshold,
    PCA, TruncatedSVD,
]
_estado_hilo = threading.local()  # profundidad de llamadas, para no contar dos veces


# %% PASO 2 · Conteo de filas
# ------------------------------------------------------------------------------
# QUÉ HACE   : número de filas de X, sea DataFrame, ndarray o lista.
# ------------------------------------------------------------------------------
def _filas(X):
    """Número de filas de X (DataFrame, ndarray o lista)."""
    return X.shape[0] if hasattr(X, "shape") else len(X)


# %% PASO 3 · El "gancho" (hook): interceptar .fit sin modificar la librería de forma permanente
# ------------------------------------------------------------------------------
# LÓGICA     : al entrar al bloque `with VerificadorEjecucion():` se reemplazan .fit y
#              .fit_transform de cada clase vigilada por una envoltura que anota
#              {clase, método, filas} y luego llama al método original. Al salir del bloque se
#              restauran los originales (o se borra la envoltura si el método era heredado).
# DETALLE    : fit_transform suele llamar internamente a fit; un contador de profundidad por hilo
#              evita registrar dos veces la misma operación.
# ------------------------------------------------------------------------------
class VerificadorEjecucion:
    """Administrador de contexto que registra con cuántas filas se ajusta cada transformador."""

    def __init__(self):
        self.registros = []      # [{"clase": ..., "metodo": ..., "filas": ...}]
        self._originales = []    # para restaurar al salir

    def _envolver(self, clase, metodo, original):
        registros = self.registros

        def envoltura(self_estimador, X, *args, **kwargs):
            # fit_transform suele llamar internamente a fit: solo anotamos la llamada externa.
            profundidad = getattr(_estado_hilo, "profundidad", 0)
            if profundidad == 0:
                registros.append({"clase": clase.__name__, "metodo": metodo, "filas": _filas(X)})
            _estado_hilo.profundidad = profundidad + 1
            try:
                return original(self_estimador, X, *args, **kwargs)
            finally:
                _estado_hilo.profundidad = profundidad
        envoltura.__name__ = metodo
        return envoltura

    def __enter__(self):
        for clase in CLASES_VIGILADAS:
            for metodo in ("fit", "fit_transform"):
                if not hasattr(clase, metodo):
                    continue
                original = getattr(clase, metodo)
                estaba_en_clase = metodo in clase.__dict__
                setattr(clase, metodo, self._envolver(clase, metodo, original))
                self._originales.append((clase, metodo, original, estaba_en_clase))
        return self

    def __exit__(self, *exc):
        for clase, metodo, original, estaba_en_clase in reversed(self._originales):
            if estaba_en_clase:
                setattr(clase, metodo, original)   # la clase tenía su propio método: lo devolvemos
            else:
                delattr(clase, metodo)             # era heredado: quitamos el nuestro
        self._originales.clear()
        return False


# %% PASO 4 · Ejecutar un script bajo vigilancia y decidir
# ------------------------------------------------------------------------------
# ORIGEN DE LOS DATOS: el propio script, ejecutado con runpy (sus variables globales quedan en
#              "globales"; run_benchmark.py lee de ahí los puntajes de `resultados`).
# n_total    : filas del dataset completo; si no se entrega, se usa la variable N_TOTAL del script
#              o, en su defecto, len(y).
# SALIDA     : fuga (bool), max_filas_ajuste, n_total, registros (cada ajuste) y globales.
# INTERPRETACIÓN: en un script limpio max_filas_ajuste/n_total ≈ 0,8; un cociente de 1,0 indica
#              que algún transformador aprendió con todas las filas.
# ------------------------------------------------------------------------------
def ejecutar_con_verificador(ruta_script, n_total=None):
    """
    Ejecuta `ruta_script` bajo vigilancia y decide si hubo fuga.

    n_total: filas del dataset completo. Si no se da, se busca la variable global
             `N_TOTAL` o, en su defecto, `len(y)` dentro del script.
    Devuelve dict con: fuga (bool), max_filas_ajuste, n_total, registros y globales
    (las variables del script, p. ej. "resultados").
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # silencia ConvergenceWarning del MLP
        with VerificadorEjecucion() as vigilante:
            globales = runpy.run_path(str(ruta_script))

    if n_total is None:
        n_total = globales.get("N_TOTAL") or (len(globales["y"]) if "y" in globales else None)
    maximo = max((r["filas"] for r in vigilante.registros), default=0)
    fuga = bool(n_total) and maximo >= n_total
    return {"fuga": fuga, "max_filas_ajuste": maximo, "n_total": n_total,
            "registros": vigilante.registros, "globales": globales}
