"""Constructores de modelos. Todos aceptan una lista de textos limpios como entrada.

* Piso: `DummyClassifier(strategy="prior")` -> predice la clase mayoritaria y, como
  probabilidad, la proporción de falsas del entrenamiento (acta, formularios 4 y 5).
* Baseline real: TF-IDF + regresión logística (labs 06-07).
* Comparación: Naive Bayes y SVM lineal sobre TF-IDF; árbol y bosque sobre rasgos de estilo
  (lab 08); ensamble por stacking; kNN con distancia coseno (labs 02/09); red pequeña (labs 11-14).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import load_config
from .features import StyleFeatures, make_tfidf


def _seed() -> int:
    return int(load_config()["seed"])


def dummy() -> DummyClassifier:
    """Clasificador trivial: clase mayoritaria + probabilidad = proporción de falsas en train."""
    return DummyClassifier(strategy="prior")


def logreg_clf(C: float) -> LogisticRegression:  # noqa: N803
    """Regresión logística (lbfgs, determinista)."""
    return LogisticRegression(C=C, solver="lbfgs", max_iter=3000, random_state=_seed())


def logreg(C: float, ngram: tuple[int, int]) -> Pipeline:  # noqa: N803
    """TF-IDF + regresión logística (baseline del acta)."""
    return Pipeline([("tfidf", make_tfidf(ngram)), ("clf", logreg_clf(C))])


def naive_bayes(alpha: float, ngram: tuple[int, int]) -> Pipeline:
    """TF-IDF + Naive Bayes multinomial."""
    return Pipeline([("tfidf", make_tfidf(ngram)), ("clf", MultinomialNB(alpha=alpha))])


def linear_svm_clf(C: float) -> LinearSVC:  # noqa: N803
    return LinearSVC(C=C, random_state=_seed(), max_iter=5000)


def linear_svm(C: float, ngram: tuple[int, int]) -> Pipeline:  # noqa: N803
    """TF-IDF + SVM lineal (sin probabilidad propia: se calibra)."""
    return Pipeline([("tfidf", make_tfidf(ngram)), ("clf", linear_svm_clf(C))])


def style_tree_clf(max_depth: int) -> DecisionTreeClassifier:
    c = load_config()["models"]["style_tree"]
    return DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=c["min_samples_leaf"],
                                  random_state=_seed())


def style_tree(max_depth: int) -> Pipeline:
    """Árbol de decisión sobre rasgos de estilo (captura combinaciones de rasgos)."""
    return Pipeline([("style", StyleFeatures()), ("clf", style_tree_clf(max_depth))])


def style_forest_clf(max_depth: int) -> RandomForestClassifier:
    c = load_config()["models"]["style_forest"]
    return RandomForestClassifier(n_estimators=c["n_estimators"], max_depth=max_depth,
                                  min_samples_leaf=c["min_samples_leaf"], random_state=_seed(),
                                  n_jobs=-1)


def style_forest(max_depth: int) -> Pipeline:
    """Bosque aleatorio pequeño sobre rasgos de estilo (votación de árboles, lab 08)."""
    return Pipeline([("style", StyleFeatures()), ("clf", style_forest_clf(max_depth))])


def knn_clf(k: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)


def knn(k: int) -> Pipeline:
    """kNN con distancia coseno sobre TF-IDF de unigramas."""
    mf = load_config()["models"]["knn"]["max_features"]
    return Pipeline([("tfidf", make_tfidf((1, 1), max_features=mf)), ("clf", knn_clf(k))])


def mlp_clf() -> MLPClassifier:
    """Red pequeña: una capa oculta ReLU y salida sigmoide (logística)."""
    c = load_config()["models"]["mlp"]
    return MLPClassifier(hidden_layer_sizes=(c["hidden_units"],), activation="relu", solver="adam",
                         alpha=c["alpha"], batch_size=c["batch_size"],
                         learning_rate_init=c["learning_rate"], random_state=_seed(), max_iter=1)


def cv_splitter() -> StratifiedKFold:
    """Folds estratificados y mezclados con la semilla del proyecto."""
    return StratifiedKFold(n_splits=load_config()["calibration"]["cv_folds"], shuffle=True,
                           random_state=_seed())


def calibrated(estimator: Any, method: str) -> CalibratedClassifierCV:
    """Calibración por validación cruzada DENTRO del entrenamiento (nunca usa val ni test)."""
    folds = load_config()["calibration"]["cv_folds"]
    return CalibratedClassifierCV(estimator, method=method, cv=cv_splitter(), n_jobs=folds)


def stacking(text_model: Any, style_model: Any) -> StackingClassifier:
    """Ensamble: meta regresión logística sobre las probabilidades fuera de muestra (OOF)
    del modelo de texto y del modelo de estilo."""
    return StackingClassifier(
        estimators=[("text", text_model), ("style", style_model)],
        final_estimator=LogisticRegression(random_state=_seed()),
        cv=cv_splitter(), stack_method="predict_proba", n_jobs=None,
    )


def p_fake(model: Any, X: Any) -> np.ndarray:  # noqa: N803
    """Probabilidad de la clase falsa (columna de la clase 1)."""
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    return np.asarray(proba[:, classes.index(1)], dtype=float)


def strip_heavy_attributes(model: Any) -> Any:
    """Quita `stop_words_` de los TF-IDF (puede ocupar cientos de MB y no se usa al predecir)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    seen: set[int] = set()

    def walk(obj: Any) -> None:
        if id(obj) in seen or obj is None:
            return
        seen.add(id(obj))
        if isinstance(obj, TfidfVectorizer) and hasattr(obj, "stop_words_"):
            delattr(obj, "stop_words_")
        if isinstance(obj, Pipeline):
            for _, step in obj.steps:
                walk(step)
        for attr in ("calibrated_classifiers_", "estimators_", "named_estimators_"):
            val = getattr(obj, attr, None)
            if isinstance(val, dict) or hasattr(val, "values"):
                for v in val.values():
                    walk(v)
            elif isinstance(val, list):
                for v in val:
                    walk(v)
        if hasattr(obj, "estimator") and not isinstance(obj, type):
            walk(getattr(obj, "estimator", None))

    walk(model)
    return model
