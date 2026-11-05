# 00 · Exploración inicial y plan

Documento del Paso 0: estado del repo, contrato de aceptación (acta), laboratorios, entorno y datos,
más las decisiones que se tomaron sin consultar y su motivo.

## 1. Repo al inicio

- Rama `dev` (existe también `main`), remoto `origin` → `github.com/devotoUCA/AI-FakeNewsDetector-2026`.
- Un solo commit (`Initial commit`) con `dummy.txt` vacío, más una carpeta vacía `AI-FakeNewsDetector-2026/`
  sin seguimiento en git. **La carpeta vacía se eliminó**; `dummy.txt` se dejó intacto.
- No había `README`, `requirements`, código ni convenciones previas: se adoptó la estructura del prompt.

## 2. Acta → checklist de aceptación

Fuente: `Datector_Fake_News-Cortina_Coronel_Denti_Devoto_Pompozzi.pdf` (11 páginas, leído completo).
No se encontraron contradicciones entre el acta y el prompt que cambien el diseño.

| # | Requisito verificable (acta) | Dónde se verifica |
|---|---|---|
| O1 | Dataset público documentado (fuente, licencia, clases) y limpio | `docs/data_card.md`, `reports/splits_manifest.json` |
| O2 | Baseline simple (TF-IDF + regresión logística) y clasificador trivial como piso | `reports/val_results.json`, `reports/test_results.json` |
| O3 | ≥ 1 modelo adicional comparado con el baseline **en el mismo test** | `reports/test_comparison.csv` |
| O4 | Accuracy, precisión, recall y F1 por clase + análisis de errores | `reports/test_results.json`, `reports/error_analysis.csv` |
| O5 | Demo: pegar texto → etiqueta + probabilidad | `app/demo.py`, `tests/test_demo.py` |
| O6 | Limitaciones y riesgos éticos | `docs/informe_resultados.md` §11 |
| E2–E7 | Entregables: dataset, notebook EDA, modelos, informe con matriz de confusión, demo, defensa | ver README |
| CE1 | Accuracy test > 80 % | `reports/success_criteria.json` |
| CE2 | F1 real y F1 falsa > 0,80 | ídem |
| CE3 | Modelo adicional ≥ baseline con la diferencia explicada | ídem + McNemar |
| CE4 | Train/val/test sin filtración | `reports/splits_manifest.json` (exactos + casi-duplicados) |
| CE5 | ≥ 20 errores revisados y comentados **por el equipo** | `reports/error_analysis.csv` (columna `comentario_equipo`) |
| CE6 | Demo devuelve etiqueta y probabilidad | tests |
| CE7 | Accuracy > 98 % se trata con sospecha | alerta automática en `evaluate` |
| R1–R7 | Riesgos y mitigaciones (rasgos superficiales, sesgo de período, etiquetas ambiguas, duplicados, plazo, errores, cómputo) | informe §11 |
| F1–F15 | Campos `[completar]` de los formularios 3, 4, 5, 8, 12, 13 y 14 | `docs/campos_completar.md` |
| F15 | Rama de visión: **no incluir** (solo mencionarla) | informe §11 |

## 3. Laboratorios

`Descargar_Practicas_IA_22_Laboratorios.zip` contiene 22 páginas HTML (`Practica_01` … `Practica_22`).
Se extrajo en `_refs/labs/` (ignorado por git) y se convirtió a texto con un parser HTML propio,
**sin ejecutar nada**. Cada lab tiene: concepto, un laboratorio en Python puro que corre en un
mini-intérprete JavaScript del navegador (sin librerías de ML; el lab 02 dice explícitamente "No vamos
a usar scikit-learn todavía"), TODOs sin resolver y un **formulario de avance del proyecto** que se
acumula en un JSON. Los formularios 1–15 coinciden uno a uno con las secciones 1–15 del acta.

**Convenciones de código de los labs:** type hints en todas las variables y funciones
(`umbral: float = 0.5`), funciones pequeñas, identificadores mayormente en español pero mezclados
(`clasificar(score, threshold)`), estilo "construcción desde cero". Como la convención de idioma no es
uniforme, se mantuvieron identificadores en inglés (regla por defecto del prompt) y se adoptaron los
type hints y las funciones pequeñas.

| Lab | Componente que prueba | Componente del proyecto | Módulo |
|---|---|---|---|
| 01 Representación | vectores, producto escalar, distancia | texto → vector TF-IDF; `dot_product`, `euclidean_distance` | `features.py`, `scratch.py` |
| 02 1-NN | clasificador por vecino más cercano desde cero | `knn_predict` (k=1 es 1-NN), comparado con sklearn | `scratch.py`, test |
| 03 Datos y calidad | faltantes, fuera de rango, **fuga** | auditoría de calidad y de fuga | `audit.py` |
| 04 Regresión y baseline | MSE, baseline de referencia | baseline trivial; Brier = MSE de la probabilidad | `models.dummy`, `metrics.py` |
| 05 Regresión aplicada | variable continua | probabilidad de falsa (0–1), Brier y log-loss | `metrics.py` |
| 06 Clasificación y umbrales | puntaje → umbral → clase; costo FP/FN | elección de umbral en validación, FPR máx. | `thresholds.py` |
| 07 Regresión logística | sigmoide, probabilidad, frontera | baseline LR TF-IDF; LR desde cero | `models.logreg`, `scratch.py` |
| 08 Random Forest / ensamble | no linealidad, votación de árboles | árbol y bosque de estilo; stacking texto+estilo; desacuerdos | `models.py`, `train.py` |
| 09 k-NN y vecindad | k vecinos, efecto de escalas | kNN coseno (k ∈ {3,5,7,11,15}); kNN de estilo con/sin estandarizar | `models.knn`, `train.py` |
| 10 K-Means, GMM, anomalías | clustering, pertenencia suave, anomalías | k-means (codo + silueta), GMM, anomalías, eje temporal | `analyze.py` |
| 11 Neurona | z = Wx + b, ReLU | MLP (ReLU oculta, sigmoide de salida) | `models.mlp_clf` |
| 12 Gradiente | actualización de pesos | descenso de gradiente de la LR desde cero | `scratch.py` |
| 13 Entrenamiento y sobreajuste | épocas, tasa de aprendizaje, brecha train/val | curvas por época (MLP y LR desde cero); brecha F1 train-val en la grilla | `train.py` |
| 14 Red vs baseline | ¿la red agrega valor? | regla del acta: adoptar la red solo si mejora en val **y** test | `evaluate.py` |
| 15 Visión | convolución | **no incluido** (acta): solo se discute en el informe | — |
| 16 NLP y embeddings | texto como vector, similitud coseno | TF-IDF, coseno, LSA (SVD) como representación densa | `features.py`, `analyze.py` |
| 17 IA generativa y RAG | recuperar antes de responder, abstención | la demo muestra las noticias de train más parecidas (trazabilidad); zona ambigua = abstenerse | `predict.similar_examples` |
| 18–19 Algoritmos genéticos | población, fitness, selección, cruce, mutación | AG para elegir subconjunto de rasgos de estilo (exploratorio) | `analyze.genetic_feature_selection` |
| 20 Riesgos y arquitectura | matriz probabilidad × impacto | matriz de riesgos del informe con la misma regla | informe §11 |
| 21 Despliegue y monitoreo | validar entradas, desvío, versionado, registro | advertencias fuera de distribución, versión de modelo, registro opcional por hash, deriva en LIAR, tests | `predict.py`, `evaluate.py` |
| 22 Integración y defensa | checklist de evidencias | `reports/success_criteria.json`, guion de defensa | `evaluate.success_criteria` |

Discrepancia menor: los labs 16–22 tienen formularios que el acta no incluye (el acta llega al 15).
No cambia el diseño: sus componentes se implementaron de forma liviana y se documentan en
`docs/mapeo_labs.md`.

## 4. Entorno

- macOS (Apple M5, 16 GB RAM), sin GPU CUDA. Python del sistema 3.14.3; se usó **Python 3.12.13** en
  `.venv/` porque tiene ruedas binarias estables para todo el stack.
- Hay internet. Stack instalado con versiones fijadas en `requirements.txt` (scikit-learn 1.9.1,
  pandas 3.1.1, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, streamlit 1.66.1, pytest 9.2.0).
- Sin torch/transformers: no hicieron falta (ver informe, decisión sobre la red).

## 5. Datos

No había datos en el repo. Ambos datasets se descargaron sin login:
- **ISOT / Fake and Real News Dataset (Kaggle)**: el endpoint público de descarga de Kaggle funcionó
  sin token. Licencia declarada en Kaggle: **CC BY-NC-SA 4.0** (fuente original: ISOT, Univ. of Victoria).
- **LIAR**: zip público de la UCSB. Licencia: el README dice "research purposes only" y que las fuentes
  originales conservan el copyright.

## 6. Plan (resumen)

1. Auditoría empírica de fuga sobre el texto crudo → reglas de limpieza justificadas con conteos.
2. Limpieza única (train = inferencia), filtro de textos cortos, deduplicación exacta y MinHash
   **antes** de dividir; splits 70/15/15 estratificados con semilla 42 y chequeo de solapamiento.
3. Baselines (trivial + LR TF-IDF con grilla C × n-gramas) y comparación (NB, SVM, árbol/bosque de
   estilo, stacking, kNN coseno, MLP) con selección **solo en validación**.
4. Calibración por CV dentro de train, umbral y cortes de confianza en validación, zona ambigua 0,4–0,6.
5. Experimento controlado de fuga (limpieza minimal vs full), validación externa en LIAR sin reentrenar.
6. Evaluación final en test una vez por modelo, con registro en `reports/test_access_log.jsonl`.
7. Análisis de errores (≥ 40 casos con comentario BORRADOR), clustering, AG exploratorio.
8. `predict()` único + CLI + demo Streamlit; tests; informe y documentación con números de `reports/`.

## 7. Decisiones tomadas sin consultar

| Decisión | Motivo |
|---|---|
| Entrada = título + "\n" + cuerpo | el acta permite título y/o cuerpo; el usuario de la demo pega cualquiera de los dos |
| `subject` y `date` nunca entran al modelo | separan las clases al 100 % (medido); el acta pide solo texto |
| Comillas y apóstrofos se unifican a espacio | la codificación tipográfica difiere por fuente (hallazgo de la auditoría) |
| Se quitan instrucciones de video embebido ("Watch the video below:") | son formato de plataforma de blog, no contenido |
| Filtro de textos < 20 palabras tras limpiar | casi todos son falsas reducidas al título: atajo "vacío ⇒ falsa" |
| Casi-duplicados: MinHash (5-shingles, 64 perm., 16 bandas), Jaccard ≥ 0,8; grupos con etiquetas mezcladas se descartan enteros | evita filtración entre splits y etiquetas contradictorias |
| Calibración elegida entre {ninguna, sigmoide, isotónica} por Brier en validación | "calibrar solo si mejora"; nunca usa test |
| Umbral: máx. F1 de falsa con FPR ≤ 10 %; empates → el más cercano a 0,5 | criterio de costos del acta sin mover el corte por ruido |
| Nivel de confianza por accuracy **local** en validación | la accuracy acumulada escondía la franja intermedia |
| Productivo = mejor F1 de falsa en validación entre modelos calibrados; empates (< 0,002) → el más simple | interpretabilidad y simplicidad (acta, formulario 14) |
| Los modelos se entrenan solo con train (no train+val) | validación queda limpia para calibración, umbral y confianza |
| Identificadores en inglés | los labs mezclan idiomas: no hay convención clara |
| Se eliminó la carpeta vacía `AI-FakeNewsDetector-2026/` | estaba vacía y sin seguimiento en git |
