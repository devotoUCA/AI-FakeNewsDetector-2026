"""Análisis exploratorios (no forman parte del modelo productivo).

* Diagnóstico del "ambiente informativo" (acta, formulario 10; lab 10): k-means sobre TF-IDF
  reducido con SVD (LSA), k elegido con codo y silueta, proporción de falsas por grupo,
  términos por grupo, pertenencia suave con GMM, anomalías y evolución temporal.
* Algoritmo genético (labs 18-19): selección del subconjunto de rasgos de estilo para el árbol.
  Se usa SOLO validación como fitness y su resultado NO cambia el modelo productivo.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer

from . import models as M
from . import plots
from .config import get_paths, load_config, write_json
from .features import STYLE_FEATURES, StyleFeatures, make_tfidf
from .splits import load_split

LOGGER = logging.getLogger(__name__)


def clustering_analysis() -> dict[str, Any]:
    """k-means + GMM sobre el conjunto de entrenamiento."""
    cfg, paths = load_config(), get_paths()
    c, seed = cfg["clustering"], cfg["seed"]
    train = load_split("train")
    texts, y = train["doc"].tolist(), train["label"].to_numpy()
    vec = make_tfidf((1, 1), max_features=30000)
    X = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=c["svd_components"], random_state=seed)
    Z = Normalizer().fit_transform(svd.fit_transform(X))
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(Z), size=min(c["silhouette_sample"], len(Z)), replace=False)
    rows, fits = [], {}
    for k in c["k_grid"]:
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(Z)
        fits[k] = km
        rows.append({"k": k, "inercia": float(km.inertia_),
                     "silueta_muestra": float(silhouette_score(Z[sample], km.labels_[sample]))})
    best_k = max(rows, key=lambda r: r["silueta_muestra"])["k"]
    km = fits[best_k]
    plots.line_plot({"Inercia (codo)": ([r["k"] for r in rows], [r["inercia"] for r in rows])},
                    "Método del codo (k-means sobre LSA)", "k", "Inercia", paths.figures / "kmeans_elbow.png")
    plots.line_plot({"Silueta": ([r["k"] for r in rows], [r["silueta_muestra"] for r in rows])},
                    "Coeficiente de silueta", "k", "Silueta (muestra)", paths.figures / "kmeans_silhouette.png")

    terms = vec.get_feature_names_out()
    centers_tfidf = svd.inverse_transform(km.cluster_centers_)
    groups = []
    for g in range(best_k):
        m = km.labels_ == g
        top = [terms[j] for j in np.argsort(-centers_tfidf[g])[:12]]
        groups.append({"grupo": g, "n": int(m.sum()), "prop_falsas": float(y[m].mean()),
                       "terminos_top": top})
    groups.sort(key=lambda d: -d["prop_falsas"])
    plots.stacked_share([f"G{d['grupo']}: {', '.join(d['terminos_top'][:3])}" for d in groups],
                        [d["prop_falsas"] for d in groups], [d["n"] for d in groups],
                        f"Proporción de falsas por grupo (k={best_k})", paths.figures / "kmeans_fake_share.png")

    dist = km.transform(Z).min(axis=1)
    cut = float(np.percentile(dist, c["anomaly_percentile"]))
    anom = np.where(dist > cut)[0]
    anomalies = [{"id": train["id"].iloc[i], "etiqueta": int(y[i]), "distancia": round(float(dist[i]), 4),
                  "titulo": texts[i].split("\n")[0][:140]} for i in anom[np.argsort(-dist[anom])][:15]]

    gmm = GaussianMixture(n_components=best_k, covariance_type="diag", random_state=seed)
    Zg = Z[:, :c["gmm_components_dims"]]
    gmm.fit(Zg)
    resp = gmm.predict_proba(Zg)
    maxr = resp.max(axis=1)

    dates = pd.to_datetime(train["date"].str.strip(), errors="coerce", format="mixed")
    period = dates.dt.to_period("Q").astype(str)
    temporal = (pd.DataFrame({"periodo": period, "grupo": km.labels_, "falsa": y})
                .query("periodo != 'NaT'").groupby("periodo")
                .agg(n=("falsa", "size"), prop_falsas=("falsa", "mean")).reset_index())
    out = {"k_elegido": best_k, "criterio": "mayor silueta (muestra); el codo se reporta como apoyo",
           "grilla": rows, "grupos": groups, "anomalias": {"percentil": c["anomaly_percentile"],
                                                          "corte_distancia": cut, "n": int(len(anom)),
                                                          "prop_falsas": float(y[anom].mean()),
                                                          "ejemplos": anomalies},
           "gmm": {"frac_pertenencia_ambigua_(<0.6)": float((maxr < 0.6).mean()),
                   "pertenencia_media": float(maxr.mean())},
           "temporal_por_trimestre": temporal.to_dict(orient="records"),
           "nota_temporal": "Las fechas no son comparables entre clases: las reales van de 2016-01 a "
                            "2017-12 y las falsas de 2015-03 a 2018-02 (ver data_card). La proporción de "
                            "falsas por trimestre refleja cómo se armó el dataset, no la evolución real "
                            "de la desinformación."}
    write_json(out, paths.reports / "clustering.json")
    return out


def genetic_feature_selection() -> dict[str, Any]:
    """Algoritmo genético: cromosoma = máscara binaria sobre los rasgos de estilo.

    Fitness = F1 de falsa en validación del árbol (profundidad elegida en train.py) menos una
    penalización por cantidad de rasgos (parsimonia). Selección por torneo, cruce de un punto y
    mutación por bit, con elitismo.
    """
    cfg, paths = load_config(), get_paths()
    g, seed = cfg["genetic"], cfg["seed"]
    rng = np.random.default_rng(seed)
    train, val = load_split("train"), load_split("val")
    S_tr, S_va = StyleFeatures().transform(train["doc"].tolist()), StyleFeatures().transform(val["doc"].tolist())
    ytr, yva = train["label"].to_numpy(), val["label"].to_numpy()
    from .config import read_json

    depth = read_json(paths.reports / "val_results.json")["seleccion"]["style_tree"]["max_depth"]
    n = len(STYLE_FEATURES)
    cache: dict[tuple[int, ...], float] = {}

    def f1_of(mask: np.ndarray) -> float:
        key = tuple(int(b) for b in mask)
        if key not in cache:
            if mask.sum() == 0:
                cache[key] = 0.0
            else:
                clf = M.style_tree_clf(depth).fit(S_tr[:, mask], ytr)
                cache[key] = float(f1_score(yva, clf.predict(S_va[:, mask])))
        return cache[key]

    def fitness(mask: np.ndarray) -> float:
        return f1_of(mask) - g["feature_penalty"] * mask.sum()

    pop = rng.random((g["population"], n)) < 0.5
    pop[0] = True  # incluye el individuo "todos los rasgos"
    history = []
    for gen in range(g["generations"]):
        fit = np.array([fitness(ind) for ind in pop])
        best_i = int(fit.argmax())
        history.append({"generacion": gen + 1, "mejor_fitness": float(fit.max()),
                        "fitness_medio": float(fit.mean()), "rasgos_mejor": int(pop[best_i].sum())})
        new = [pop[best_i].copy()]  # elitismo
        while len(new) < len(pop):
            a, b = (pop[max(rng.choice(len(pop), 3, replace=False), key=lambda i: fit[i])] for _ in range(2))
            cut = rng.integers(1, n)
            child = np.r_[a[:cut], b[cut:]]
            flip = rng.random(n) < g["mutation_rate"]
            child = np.where(flip, ~child, child)
            new.append(child)
        pop = np.array(new)
    fit = np.array([fitness(ind) for ind in pop])
    best = pop[int(fit.argmax())]
    all_mask = np.ones(n, dtype=bool)
    out = {"profundidad_arbol": depth, "historia": history,
           "mejor_subconjunto": [f for f, b in zip(STYLE_FEATURES, best) if b],
           "f1_val_mejor_subconjunto": f1_of(best), "f1_val_todos_los_rasgos": f1_of(all_mask),
           "evaluaciones_distintas": len(cache),
           "nota": "Exploratorio: el fitness usa validación, así que el F1 del subconjunto es "
                   "optimista. No se usa para el modelo productivo ni se evalúa en test."}
    plots.line_plot({"Mejor": ([h["generacion"] for h in history], [h["mejor_fitness"] for h in history]),
                     "Promedio": ([h["generacion"] for h in history], [h["fitness_medio"] for h in history])},
                    "Algoritmo genético: fitness por generación", "Generación", "Fitness",
                    paths.figures / "genetic_fitness.png")
    write_json(out, paths.reports / "genetic_feature_selection.json")
    return out
