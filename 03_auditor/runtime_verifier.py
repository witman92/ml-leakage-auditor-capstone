"""
runtime_verifier.py  ·  Verificador en ejecución (segunda opinión del auditor).

IDEA EN UNA FRASE
    Ejecuta el script de verdad y cuenta con cuántas FILAS se ajusta cada transformador.
    Si alguno se ajusta con tantas filas como tiene el dataset completo, vio el conjunto
    de prueba: hay fuga.

¿POR QUÉ HACE FALTA SI YA HAY UN AUDITOR AST?
    El AST lee el código, pero no lo entiende del todo (nombres dinámicos, funciones
    auxiliares...). El verificador mide lo que PASÓ, no lo que el código parece decir.
    Son dos mecanismos distintos: si coinciden, la conclusión es más sólida.

CÓMO FUNCIONA (técnica de "gancho" o hook)
    Mientras dura el bloque `with VerificadorEjecucion():` se reemplazan los métodos
    .fit y .fit_transform de los transformadores de scikit-learn por versiones que,
    antes de llamar al original, anotan cuántas filas recibieron. Al salir del bloque
    se restauran los métodos originales: no queda ninguna modificación permanente.

    Regla de decisión:   n_filas_ajuste >= n_total   =>   FUGA
    (en validación cruzada de 5 pliegues un transformador bien puesto ve ~80 % de las filas)

USO
    from runtime_verifier import ejecutar_con_verificador
    r = ejecutar_con_verificador("mi_script.py")
    r["fuga"], r["max_filas_ajuste"], r["n_total"], r["globales"]["resultados"]
"""

import runpy
import threading
import warnings

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import (RFE, SelectFromModel, SelectKBest, SelectPercentile,
                                       VarianceThreshold)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, PowerTransformer,
                                   QuantileTransformer, RobustScaler, StandardScaler)

CLASES_VIGILADAS = [
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer,
    SimpleImputer, KNNImputer,
    SelectKBest, SelectPercentile, SelectFromModel, RFE, VarianceThreshold,
    PCA, TruncatedSVD,
]
_estado_hilo = threading.local()  # profundidad de llamadas, para no contar dos veces


def _filas(X):
    """Número de filas de X (DataFrame, ndarray o lista)."""
    return X.shape[0] if hasattr(X, "shape") else len(X)


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
