# Mapeo laboratorios → componente → módulo (estado final)

Los labs son ejercicios de Python puro, sin resolver. Acá se indica dónde quedó implementado cada
componente en el proyecto y cómo se verifica. La columna "Evidencia" apunta a un test o a un reporte
real generado en la corrida.

| Lab | Componente | Implementación en el proyecto | Evidencia |
|---|---|---|---|
| 01 Representación del problema | texto → vector; producto escalar, distancia | `features.make_tfidf`; `scratch.dot_product`, `scratch.euclidean_distance` | `tests/test_scratch.py::test_vector_ops` |
| 02 Primer modelo 1-NN | vecino más cercano desde cero | `scratch.knn_predict` (con k=1 es 1-NN) | coincide con sklearn: `test_knn_matches_sklearn` |
| 03 Datos y calidad | faltantes, atípicos, fuga | `audit.data_quality`, `audit.marker_table`, metadatos solos; filtro de cortos y deduplicación en `splits.py` | `reports/data_audit.json`, `leakage_audit.json` |
| 04 Regresión y baseline | baseline de referencia, MSE | `models.dummy` (prior); Brier = MSE de la probabilidad | `test_results.json` (Brier del trivial 0,248) |
| 05 Regresión aplicada | variable continua documentada | probabilidad de falsa: Brier, log-loss, ECE | `metrics.probability_metrics` |
| 06 Clasificación y umbrales | puntaje → umbral; costo de FP/FN | `thresholds.choose_threshold` (máx. F1 con FPR ≤ 10 %), `decide` | `decision_val.json`, `pr_curve_val.png` |
| 07 Regresión logística | sigmoide y frontera de decisión | baseline `models.logreg`; `scratch.sigmoid` | `baseline_logreg_grid.csv` |
| 08 Random Forest / ensamble | no linealidad, votación | `models.style_tree`, `style_forest`, `stacking`; `train.disagreement_analysis` | `disagreement_val.csv`; McNemar en `test_results.json` |
| 09 k-NN y vecindad | k vecinos, escalas | `models.knn` (coseno, k ∈ {3,5,7,11,15}); kNN de estilo sin y con estandarizar | `val_results.json → knn_estilo_escala` (F1 0,836 sin estandarizar, 0,909 estandarizado) |
| 10 K-Means, GMM, anomalías | grupos, pertenencia suave, anomalías | `analyze.clustering_analysis` (codo, silueta, GMM, anomalías, trimestres); `scratch.kmeans_step` | `clustering.json`, `kmeans_*.png` |
| 11 Neurona artificial | z = Wx + b, ReLU | `models.mlp_clf` (ReLU oculta, salida sigmoide); `scratch.relu` | `test_sigmoid_relu` |
| 12 Aprendizaje y gradiente | actualización por gradiente | `scratch.ScratchLogisticRegression.fit` | `scratch_vs_sklearn.json` (97,9 % de acuerdo con sklearn) |
| 13 Entrenamiento y sobreajuste | épocas, tasa, brecha train/val | curvas por época del MLP y de la LR desde cero; brecha en la grilla | `mlp_learning_curve.png`, `scratch_learning_rates.png` |
| 14 Red vs baseline | ¿la red agrega valor? | regla del acta en `evaluate.evaluate_final → decision_red` | `adoptar: false` |
| 15 Visión y convolución | — | **no incluido** por decisión del acta; se discute en el informe §12 | — |
| 16 NLP y embeddings | similitud coseno, representación densa | TF-IDF L2 + coseno; LSA (TruncatedSVD) en clustering | `test_tfidf_shape_and_determinism` |
| 17 IA generativa y RAG | recuperar con fuente, abstenerse | `Predictor.similar_examples` (3 noticias de train más parecidas en la demo); zona ambigua | `test_contract_on_article` |
| 18–19 Algoritmos genéticos | cromosoma, fitness, selección, cruce, mutación, elitismo | `analyze.genetic_feature_selection` (subconjunto de rasgos de estilo) | `genetic_feature_selection.json`, `genetic_fitness.png` |
| 20 Riesgos y arquitectura | matriz probabilidad × impacto | tabla de riesgos del informe §12 (misma regla: ≥ 16 ALTO, ≥ 8 MEDIO) | informe |
| 21 Despliegue, pruebas y monitoreo | contrato de entrada/salida, validación, deriva, versión, registro | `Predictor.input_warnings`; `metadata.json` (versión y hashes); registro opcional por hash (`demo.log_predictions`); deriva en LIAR | `tests/test_predict.py`, `external_liar.json → deriva_y_fuera_de_distribucion` |
| 22 Integración y defensa final | checklist de evidencias | `evaluate.success_criteria` → `success_criteria.json`; `docs/guion_defensa.md` | `success_criteria.json` |

## Iteración 2: componentes nuevos

| Lab | Componente | Implementación | Evidencia |
|---|---|---|---|
| 03 Datos y calidad / fuga | auditoría de atajos de un solo rasgo (AUC y stump), crudo y por perfil | `shortcuts.py` | `reports/shortcut_audit_v2.csv`, `shortcut_audit_v2.png` |
| 03 / 16 | limpieza "neutral a la fuente" (v2, v3) y verificación de etiquetas de un dataset nuevo | `cleaning.py` (perfiles), `corpus.verify_welfake_label_direction` | `tests/test_cleaning_v2.py`, `welfake_manifest_v2.json` |
| 08 Ensamble | ensamble con y sin rasgos de estilo en el panel | `panel.build_model("ensemble")` | `panel_v2.csv` |
| 09 / 16 | n-gramas de caracteres; similitud coseno como rasgo OOD | `panel.build_model("lr_char")`, `ood.ood_features` | `train_v2.json` |
| 13 Sobreajuste / generalización | panel con cambio de grupo y de tiempo, regularización más fuerte | `panel.py` | `panel_v2.csv`, `ablation_v2.csv` |
| 17 RAG: abstención | abstención por OOD, contracción hacia la prior | `ood.py`, `Predictor.predict` | `abstention_coverage_v2.png`, `tests/test_ood_v2.py` |
| 21 Monitoreo y pruebas | pruebas metamórficas de invarianza; detección de entradas fuera de distribución; versión del modelo y del perfil en la metadata | `invariance.py`, `models/metadata_v2.json` | `tests/test_invariance_v2.py` |
| 22 Defensa | criterios reevaluados; comparación iteración 1 frente a 2 | `evaluate_v2.success_criteria_v2`, `comparison_table` | `success_criteria_v2.json`, `comparison_it1_vs_it2.csv` |

## Iteración 3: componentes nuevos

| Lab | Componente | Implementación | Evidencia |
|---|---|---|---|
| 03 Datos y calidad | descubrimiento automático de atajos de formato; verificación de un dataset nuevo | `format_shortcuts.py`, `corpus.prepare_fakes` | `format_shortcut_discovery_v3.csv`, `fakes_manifest_v3.json` |
| 05 / 07 Probabilidad | probabilidad calibrada sin contraer; extremos como "< 1 %" | `predict.clip_public`, `display_probability` | `shrink_resolution_v3.json` |
| 16 NLP y embeddings | embeddings congelados (MiniLM) como experimento opcional | `embeddings.py`, `requirements-extra.txt` | `panel_v3.csv` (resultado negativo) |
| 17 RAG: abstención | sin veredicto fuera del dominio de validez | `domain.py`, `Predictor._predict_with_gate` | `domain_gate_v3.json`, `tests/test_predict_v3.py` |
| 20 Riesgos | matriz de riesgos actualizada | informe §8 | informe |
| 21 Monitoreo y pruebas | invarianza con IC; log de accesos verificable; CI; runner multiplataforma | `invariance.flip_stats_ci`, `access_log.py`, `.github/workflows/ci.yml`, `tasks.py` | `invariance_v3.json`, `tests/test_reports_v3.py` |
| 22 Defensa | model card, tablas generadas desde reportes, cierre de la alerta | `docs/model_card.md`, `make_report.py` | informe §9–10 |
