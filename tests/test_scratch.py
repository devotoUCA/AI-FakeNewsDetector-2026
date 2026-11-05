"""Las implementaciones desde cero (estilo labs) coinciden con numpy/scikit-learn."""

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from fakenews.scratch import (ScratchLogisticRegression, cosine_similarity, dot_product,
                              euclidean_distance, kmeans_step, knn_predict, relu, sigmoid)


def test_vector_ops():
    a, b = [22.0, 70.0], [31.0, 38.0]
    assert dot_product(a, b) == np.dot(a, b)
    assert np.isclose(euclidean_distance(a, b), np.linalg.norm(np.subtract(a, b)))
    assert np.isclose(cosine_similarity([1, 1, 0, 0], [1, 1, 1, 0]), 2 / (np.sqrt(2) * np.sqrt(3)))
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_knn_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.random((40, 5))
    y = (X[:, 0] > 0.5).astype(int)
    q = rng.random((10, 5))
    sk = KNeighborsClassifier(5, metric="cosine", algorithm="brute").fit(X, y).predict(q)
    mine = [knn_predict(list(r), X.tolist(), y.tolist(), 5) for r in q]
    assert list(sk) == mine


def test_sigmoid_relu():
    assert sigmoid(0.0) == 0.5
    assert np.allclose(sigmoid(np.array([-800.0, 800.0])), [0.0, 1.0])
    assert relu(-1.0) == 0.0 and relu(2.0) == 2.0


def test_kmeans_step_matches_sklearn_single_iteration():
    pts = np.array([[21.0, 66.0], [22.0, 70.0], [24.0, 64.0], [31.0, 47.0], [33.0, 43.0], [35.0, 50.0]])
    c0 = np.array([[22.0, 68.0], [33.0, 46.0]])
    assign, centers = kmeans_step(pts, c0)
    km = KMeans(2, init=c0, n_init=1, max_iter=1).fit(pts)
    assert np.allclose(np.sort(centers, axis=0), np.sort(km.cluster_centers_, axis=0))
    assert list(assign) == [0, 0, 0, 1, 1, 1]


def test_scratch_logreg_learns_and_agrees_with_sklearn():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(400, 4))
    y = (X @ np.array([2.0, -1.0, 0.5, 0.0]) + 0.2 > 0).astype(int)
    m = ScratchLogisticRegression(learning_rate=1.0, epochs=300, l2=0.0).fit(sparse.csr_matrix(X), y)
    sk = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    agree = (m.predict(sparse.csr_matrix(X)) == sk.predict(X)).mean()
    assert agree > 0.97
    losses = [h["loss_train"] for h in m.history]
    assert losses[-1] < losses[0]  # la pérdida baja con las épocas (lab 13)
