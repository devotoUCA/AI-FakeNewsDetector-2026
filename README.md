# Detector de Fake News

Clasificador de noticias **en inglés** que, dado el texto (título y/o cuerpo), devuelve:

- una etiqueta: *probablemente real*, *probablemente falsa* o *incierta, requiere revisión humana*;
- la probabilidad de que sea falsa;
- un nivel de confianza;
- los términos que más pesaron en la decisión.

Proyecto de la materia **Inteligencia Artificial (UCA, 2026)**. Docentes: Germán Lamartine y Javier
Adolfo Ouret. Equipo: Tomás Cortina, Martín Coronel, Mateo Denti, Mateo Devoto y Fabricio Pompozzi.

> ⚠️ El resultado es una estimación basada en patrones del texto, **no un veredicto ni una verificación
> de hechos**. Dentro de su dataset el modelo acierta el 98,9 %, pero en un dataset externo (LIAR)
> no supera al azar: aprendió el estilo de las fuentes de entrenamiento. Ver *Limitaciones*.

## Instalación

Requiere Python 3.10 o superior (probado con 3.12.13 en macOS arm64).

```bash
make setup                 # crea .venv, instala requirements.txt (versiones fijadas) y el paquete
```

Sin `make`:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
```

## Uso

| Comando | Qué hace |
|---|---|
| `make data` | Descarga ISOT (Kaggle, endpoint público) y LIAR, limpia, deduplica, arma los splits y corre la auditoría |
| `make train` | Entrena y selecciona todos los modelos. Las decisiones se toman **solo con validación** |
| `make evaluate` | Evaluación final en test (cada acceso queda en `reports/test_access_log.jsonl`), validación externa en LIAR, análisis de errores y criterios de éxito |
| `make analyze` | Clustering (k-means/GMM) y algoritmo genético. Son exploratorios |
| `make notebook` | Ejecuta `notebooks/01_eda_y_preparacion.ipynb` de punta a punta |
| `make test` | Suite de tests (`make test-fast` saltea el notebook) |
| `make demo` | `streamlit run app/demo.py`. Funciona offline una vez entrenado |
| `make all` | Todo lo anterior en orden |

Desde la CLI:

```bash
.venv/bin/python -m fakenews.cli predict "Paste an English news text here..." --similar
```

Desde Python: `from fakenews.predict import predict; predict(text)` devuelve un diccionario con
`label`, `p_false`, `confidence_level`, `ambiguous`, `top_terms` y `warnings`.

Tiempos de referencia (Apple M5): `data` unos 3 min, `train` unos 3 min, `evaluate` unos 40 s,
`analyze` unos 2 min.

## Resultados (test, evaluado una sola vez por modelo)

Fuente: `reports/test_comparison.csv`. IC 95 % por bootstrap.

| Modelo | Accuracy | F1 falsa [IC 95 %] | F1 real | Brier |
|---|---|---|---|---|
| Trivial (clase mayoritaria) | 0,547 | 0,000 | 0,707 | 0,248 |
| **Baseline:** TF-IDF + regresión logística | 0,986 | 0,985 [0,981–0,988] | 0,988 | 0,0104 |
| Naive Bayes | 0,947 | 0,941 | 0,951 | 0,0408 |
| **SVM lineal calibrado (productivo)** | **0,989** | **0,988 [0,985–0,991]** | **0,990** | **0,0082** |
| Árbol de estilo | 0,900 | 0,890 | 0,908 | 0,0743 |
| Bosque de estilo | 0,927 | 0,919 | 0,933 | 0,0555 |
| Ensamble (texto + estilo) | 0,987 | 0,986 | 0,989 | 0,0096 |
| kNN coseno (k = 15) | 0,900 | 0,880 | 0,914 | 0,0744 |
| Red pequeña (MLP) | 0,986 | 0,985 | 0,988 | 0,0106 |

- **SVM frente al baseline**: McNemar p = 0,030. La red no mejora, así que no se adopta.
- **Experimento de fuga**: F1 de falsa de 0,993 con las marcas de fuente y 0,985 sin ellas.
- **Validación externa (LIAR, sin reentrenar)**: accuracy 0,487, contra 0,560 del trivial.

Detalle completo en [docs/informe_resultados.md](docs/informe_resultados.md).

## Estructura

```
config.yaml                 parámetros únicos (semilla 42, rutas, grillas, umbrales)
src/fakenews/               paquete
  cleaning.py               limpieza única (train = inferencia), idempotente
  data.py, dedup.py, splits.py, audit.py     datos, MinHash, splits, auditoría de fuga
  features.py               TF-IDF y rasgos de estilo
  models.py                 trivial, LR, NB, SVM, árbol/bosque, stacking, kNN, MLP
  metrics.py, thresholds.py métricas, bootstrap, McNemar, umbral, zona ambigua, confianza
  explain.py                términos influyentes
  train.py, evaluate.py     entrenamiento (validación) y evaluación final (test, LIAR)
  analyze.py                clustering y algoritmo genético (exploratorio)
  scratch.py                componentes "desde cero" de los laboratorios
  predict.py, demo_logic.py, cli.py   inferencia, lógica de la demo, CLI
app/demo.py                 demo Streamlit
notebooks/                  EDA (entregable 3)
docs/                       exploración, data card, informe, mapeo de labs, campos del acta, guion
reports/                    métricas (JSON/CSV), figuras, registro de accesos al test
models/metadata.json        versión, hashes de datos, semilla, librerías (los .joblib no se versionan)
tests/                      pytest
```

## Limitaciones

- El modelo distingue **el estilo de Reuters del estilo de los blogs** del dataset ISOT. No distingue
  verdad de mentira: en LIAR rinde por debajo del azar.
- Solo inglés. Solo política de EE.UU. entre 2015 y 2018. Falla con sátira, con noticias nuevas y
  con temas fuera del dataset.
- Una confianza "alta" significa parecido con el entrenamiento, no certeza.
- **Nunca debe usarse como único criterio para moderar o censurar contenido.**

## Documentación

- [docs/00_exploracion.md](docs/00_exploracion.md): exploración, checklist del acta, labs, decisiones.
- [docs/data_card.md](docs/data_card.md): fuentes, licencias, conteos, fugas, sesgos.
- [docs/informe_resultados.md](docs/informe_resultados.md): informe de resultados (entregable 5).
- [docs/campos_completar.md](docs/campos_completar.md): valores reales para los `[completar]` del acta.
- [docs/mapeo_labs.md](docs/mapeo_labs.md): laboratorio → componente → módulo.
- [docs/guion_defensa.md](docs/guion_defensa.md): guion de la presentación final.

## Datos y licencias

- ISOT / *Fake and Real News Dataset* (Kaggle): CC BY-NC-SA 4.0.
- LIAR (W. Y. Wang, ACL 2017): solo para investigación.

Los datos no se incluyen en el repo; se descargan con `make data`.
