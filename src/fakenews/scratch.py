"""Implementaciones "desde cero" de los componentes de los laboratorios, aplicadas al proyecto.

Los labs construyen cada pieza sin librerías de ML (producto escalar, distancia, 1-NN, k-NN,
sigmoide, neurona, gradiente, épocas). Acá están las mismas piezas en versión vectorizada
para poder correrlas sobre TF-IDF real, y se verifican contra scikit-learn en los tests y
en `reports/scratch_vs_sklearn.json`. No forman parte del modelo productivo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse


def dot_product(a: list[float], b: list[float]) -> float:
    """Producto escalar (lab 01)."""
    suma: float = 0.0
    for x, y in zip(a, b, strict=True):
        suma = suma + x * y
    return suma


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Distancia euclidiana (labs 01-02)."""
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similitud coseno (lab 16): no depende del largo del texto."""
    den = (dot_product(a, a) ** 0.5) * (dot_product(b, b) ** 0.5)
    return 0.0 if den == 0.0 else dot_product(a, b) / den


def knn_predict(query: list[float], data: list[list[float]], labels: list[int], k: int) -> int:
    """k-NN por votación con similitud coseno (labs 02 y 09). Con k=1 es el 1-NN del lab 02."""
    sims = sorted(((cosine_similarity(query, row), i) for i, row in enumerate(data)), reverse=True)
    votes = sum(labels[i] for _, i in sims[:k])
    return 1 if votes > k / 2 else 0


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Sigmoide numéricamente estable (labs 07 y 11)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out if out.shape else float(out)


def relu(z: np.ndarray | float) -> np.ndarray | float:
    """ReLU (lab 11)."""
    return np.maximum(0.0, z)


def kmeans_step(points: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Una iteración de k-means (lab 10): asignar al centro más cercano y recalcular centros."""
    d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = d2.argmin(axis=1)
    new = np.array([points[assign == c].mean(axis=0) if np.any(assign == c) else centers[c]
                    for c in range(len(centers))])
    return assign, new


class ScratchLogisticRegression:
    """Regresión logística = una neurona con salida sigmoide, entrenada por descenso de gradiente
    (labs 07, 11, 12 y 13). Registra la pérdida por época para estudiar la tasa de aprendizaje."""

    def __init__(self, learning_rate: float = 0.5, epochs: int = 30, l2: float = 1e-4) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.history: list[dict[str, float]] = []

    @staticmethod
    def _loss(y: np.ndarray, p: np.ndarray) -> float:
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, X: sparse.spmatrix | np.ndarray, y: np.ndarray,  # noqa: N803
            X_val: Any = None, y_val: np.ndarray | None = None) -> "ScratchLogisticRegression":  # noqa: N803
        n, d = X.shape
        y = np.asarray(y, dtype=float)
        self.w = np.zeros(d)
        self.b = 0.0
        self.history = []
        for epoca in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            err = p - y                                  # dL/dz de la log-loss
            grad_w = (X.T @ err) / n + self.l2 * self.w  # gradiente promedio + L2
            grad_b = float(err.mean())
            self.w = self.w - self.learning_rate * np.asarray(grad_w).ravel()
            self.b = self.b - self.learning_rate * grad_b
            row = {"epoca": epoca + 1, "loss_train": self._loss(y, sigmoid(X @ self.w + self.b))}
            if X_val is not None and y_val is not None:
                row["loss_val"] = self._loss(np.asarray(y_val, float), self.predict_proba(X_val))
            self.history.append(row)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:  # noqa: N803
        """Probabilidad de la clase falsa."""
        if self.w is None:
            raise RuntimeError("Modelo sin entrenar.")
        return np.asarray(sigmoid(X @ self.w + self.b))

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:  # noqa: N803
        return (self.predict_proba(X) >= threshold).astype(int)
