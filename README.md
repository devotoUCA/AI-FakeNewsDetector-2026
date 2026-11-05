# Detector de Fake News

Proyecto de la materia **Inteligencia Artificial (UCA, 2026)**. Docentes: Germán Lamartine y Javier
Adolfo Ouret. Equipo: Tomás Cortina, Martín Coronel, Mateo Denti, Mateo Devoto y Fabricio Pompozzi.

## 1. Qué es y qué NO es

**Es** un clasificador binario de noticias **en inglés**, que usa **solo el texto** (título y/o
cuerpo). Para cada texto devuelve:

- una etiqueta: *probablemente real*, *probablemente falsa* o *incierta, requiere revisión humana*;
- la **probabilidad de que sea falsa**;
- un nivel de confianza;
- advertencias si la entrada no se parece a los datos de entrenamiento;
- los términos que más pesaron en la decisión.

**No es** un verificador de hechos ni un producto para moderar contenido real.

> **Hallazgo principal.** El modelo aprende sobre todo **patrones de redacción de las fuentes** del
> dataset de entrenamiento (estilo de agencia frente a estilo de blog). No aprende la veracidad. Dentro
> de su dataset acierta el 99 %; en artículos de otros medios queda al nivel de un clasificador
> trivial. Detalle en el [informe de resultados](docs/informe_resultados.md), §15.

## 2. Requisitos

| | |
|---|---|
| Python | 3.10 a 3.12. Probado con **3.12.13**. Python 3.14 no se probó. |
| Sistema operativo | **macOS (arm64): probado.** Linux debería funcionar igual. **Windows**: abajo se dan los comandos equivalentes en PowerShell, pero **no se probaron en Windows**. |
| Disco | Unos 0,9 GB para el entorno virtual. Inicio rápido: unos 0,5 GB más (datos y modelo). Pipeline completo: unos 2,3 GB más (datos crudos 0,4 GB, procesados 0,4 GB, cachés 0,4 GB, modelos 0,2 GB). |
| Memoria | 8 GB alcanzan para el inicio rápido. Para el panel de la iteración 2 (3 procesos en paralelo) se recomiendan 16 GB, o correrlo con `--jobs 1`. |
| Internet | Solo para instalar dependencias y descargar los datos. Una vez entrenado, todo (CLI y demo) funciona **offline**. |

Tiempos medidos en un Apple M5 (10 núcleos, 16 GB):

| Paso | Tiempo |
|---|---|
| `make setup` | ~20 s con caché de pip; más la primera vez (descarga ~0,9 GB de paquetes) |
| `make quickstart` | ~7 min (medido: 6 min 39 s en un entorno limpio) |
| Pipeline completo de la iteración 1 (`data` + `train` + `evaluate` + `analyze` + `notebook`) | ~12 min |
| `data-external` | ~10 min (la descarga de WELFake, 245 MB, depende de la red) |
| `shortcuts` | ~6 min la primera vez |
| `panel` | ~30 min |
| `train-v2` | ~7 min |
| `evaluate-v2` | ~4 min |

## 3. Instalación paso a paso

**macOS / Linux:**

```bash
git clone https://github.com/devotoUCA/AI-FakeNewsDetector-2026.git
cd AI-FakeNewsDetector-2026
make setup                       # crea .venv e instala requirements.txt (versiones fijadas) + el paquete
source .venv/bin/activate        # opcional: para escribir `python` en vez de `.venv/bin/python`
python -m fakenews.cli --help    # verificación ("smoke check"): debe listar los comandos
```

Sin `make`:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/python -m fakenews.cli --help
```

**Windows (PowerShell; equivalentes no probados):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # si falla: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m fakenews.cli --help
```

En Windows no hay `make`: cada target de abajo indica el comando `python -m fakenews.cli ...` que lo
reemplaza.

## 4. Datos

Se descargan automáticamente. Los archivos y licencias:

| Dataset | Uso | Descarga automática | Alternativa manual | Licencia |
|---|---|---|---|---|
| ISOT / *Fake and Real News* (Kaggle, `clmentbisaillon/fake-and-real-news-dataset`) | entrenamiento | `python -m fakenews.cli download` (endpoint público de Kaggle, sin login) | bajar de https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset y dejar `Fake.csv` y `True.csv` en `data/raw/isot_kaggle/` | **CC BY-NC-SA 4.0** según declara Kaggle. **Pendiente** confirmar las condiciones del ISOT original (University of Victoria). |
| LIAR (W. Y. Wang, 2017) | externo y D1 | mismo comando `download` | descomprimir https://www.cs.ucsb.edu/~william/data/liar_dataset.zip en `data/raw/liar/` | **solo para investigación**; las fuentes conservan el copyright |
| WELFake (Zenodo 4561253) | externo de la iteración 2 (solo evaluación) | `python -m fakenews.cli download-external` | bajar `WELFake_Dataset.csv` de https://zenodo.org/records/4561253 a `data/raw/welfake/` | **CC BY 4.0** |

Los SHA-256 esperados están en `config.yaml` (sección `download`). **Uso académico no comercial.** Los
datos no se versionan en el repo.

## 5. Inicio rápido (~7 minutos)

Descarga ISOT y LIAR, prepara los datos y entrena **solo el modelo productivo** (sin el panel ni las
comparaciones):

```bash
make quickstart                  # Windows: python -m fakenews.cli quickstart
.venv/bin/python -m fakenews.cli predict - < examples/01_agencia_real_val.txt
```

El inicio rápido reutiliza los parámetros de abstención ya calibrados que vienen en el repo
(`models/ood_params_v2.json`). Produce el mismo modelo que el pipeline completo (misma configuración
y semilla), pero sin recalcular los diagnósticos.

**Usar un modelo ya entrenado** (si el equipo lo comparte por fuera de git): copiar
`final_model_v2.joblib` y `retrieval_v2.joblib` en `models/`. `predict` y la demo los toman
automáticamente (`config.yaml → inference.artifact`). Los `.joblib` pesan ~40 MB y no se versionan.

## 6. Pipeline completo

```bash
make setup && make all
```

| Target | Comando sin `make` | Qué hace | Archivos que genera |
|---|---|---|---|
| `data` | `download`, `prepare`, `audit` | descarga, limpia, deduplica, arma los splits y audita fugas | `data/processed/*.csv`, `reports/splits_manifest.json`, `data_audit.json`, `leakage_audit.json` |
| `train` | `train` | iteración 1: baseline y modelos de comparación (solo validación) | `models/final_model.joblib`, `models/candidates/`, `reports/val_*.json` |
| `evaluate` | `evaluate` | iteración 1: test (registrado), LIAR, errores | `reports/test_results.json`, `external_liar.json`, `error_analysis.csv` |
| `data-external` | `download-external`, `prepare-external` | WELFake: verifica etiquetas, deduplica contra ISOT, divide en dev y holdout | `data/processed/welfake_*.csv`, `reports/welfake_manifest_v2.json` |
| `shortcuts` | `shortcuts` | auditoría de atajos de un solo rasgo | `reports/shortcut_audit_v2.csv`, `shortcut_tokens_v2.csv` |
| `panel` | `panel` | panel de robustez (grupo, tiempo, externo, invarianza) y selección | `reports/panel_v2.csv`, `panel_v2.json` |
| `train-v2` | `train-v2` | modelo final v2 con OOD y abstención | `models/final_model_v2.joblib`, `ood_params_v2.json`, `reports/train_v2.json` |
| `evaluate-v2` | `evaluate-v2` | evaluación final única: ISOT test + WELFake-holdout (registrada) | `reports/final_v2.json`, `comparison_it1_vs_it2.csv`, `error_analysis_v2.csv`, `success_criteria_v2.json` |
| `analyze` | `analyze` | clustering y algoritmo genético (exploratorios) | `reports/clustering.json`, `genetic_feature_selection.json` |
| `notebook` | `jupyter nbconvert --execute ...` | EDA de punta a punta | `notebooks/01_eda_y_preparacion.ipynb` |
| `test` | `pytest` | suite completa | — |
| `demo` | `streamlit run app/demo.py` | demo web local | — |

Cada comando sin `make` se corre como `python -m fakenews.cli <comando>`. Las figuras quedan en
`reports/figures/`.

## 7. Cómo probar el modelo

### Demo

```bash
make demo                        # o: .venv/bin/streamlit run app/demo.py
```

Abrir http://localhost:8501. Se pega un texto en inglés (o se elige uno de los ejemplos del dataset
de test, marcados como tales) y se aprieta **Analizar**. La pantalla muestra:

- la etiqueta;
- la probabilidad de falsa (número y barra);
- el nivel de confianza;
- el aviso de zona ambigua o fuera de distribución;
- las advertencias de entrada;
- los términos que empujaron hacia "falsa" o hacia "real";
- las 3 noticias de entrenamiento más parecidas;
- en "Detalles técnicos": la probabilidad cruda, el puntaje OOD y el perfil de limpieza.

### CLI

```bash
.venv/bin/python -m fakenews.cli predict "texto en inglés..."            # desde argumento
.venv/bin/python -m fakenews.cli predict - < examples/01_agencia_real_val.txt   # desde archivo / stdin
.venv/bin/python -m fakenews.cli predict - --similar < archivo.txt       # + noticias similares
```

Salida real (recortada) para `examples/01_agencia_real_val.txt`:

```json
{
  "label": "probablemente real",
  "p_false": 0.1526,
  "confidence_level": "media",
  "confidence": 0.8474,
  "ambiguous": false,
  "top_terms": {"hacia_falsa": [{"termino": "this ", "peso": ...}, ...],
                "hacia_real": [{"termino": " said", "peso": ...}, ...]},
  "warnings": [],
  "p_false_raw": 0.0,
  "ood_score": 0.3366,
  "cleaning_profile": "v2",
  "model_version": "v2-lrchar_v2_C10-..."
}
```

### Casos de prueba sugeridos (resultados reales con el modelo v2)

Los textos de `examples/` salen del split de **validación** de ISOT y de LIAR-valid (ver
`examples/README.md`). No son noticias inventadas.

| Caso | Entrada | Resultado obtenido |
|---|---|---|
| Noticia tipo agencia (real en el dataset) | `examples/01_agencia_real_val.txt` | *probablemente real*, p = 0,153 (cruda 0,000), confianza **media**, OOD 0,34 |
| Tono sensacionalista (falsa en el dataset) | `examples/02_sensacionalista_falsa_val.txt` | *probablemente falsa*, p = 0,641 (cruda 0,997), confianza **media**, OOD 0,65 |
| Afirmación tipo LIAR (pants-fire) | `examples/03_afirmacion_liar_valid.txt` | *probablemente falsa*, p = 0,615, confianza **media**, OOD 0,50 |
| Texto muy corto | `"Breaking news today"` | *incierta*, confianza baja, advertencia "Texto muy corto (3 palabras…)" |
| Texto en español | `"El gobierno anunció hoy un nuevo plan económico que incluye una reducción de impuestos para las pequeñas empresas y un aumento del gasto en infraestructura."` | *incierta*, advertencias "no parece estar en inglés" y "Fuera de distribución (puntaje OOD 1.00 ≥ 0.80)" |
| Solo URL | `"https://example.com/noticia"` | *incierta*, "El texto solo contiene enlaces" |
| Vacío | `""` | *incierta*, `p_false = null`, "El texto está vacío." |

### Cómo interpretar la salida

- **`label`**: *incierta* si la probabilidad cae en la **zona ambigua 0,4–0,6**, si la entrada está
  **fuera de distribución** (puntaje OOD ≥ 0,80) o si hay advertencias graves (vacío, muy corto, no
  inglés, vocabulario desconocido). Si no, compara con el umbral 0,50.
- **`p_false`**: probabilidad de falsa ya **contraída hacia la proporción base (0,453)** según el
  puntaje OOD. `p_false_raw` es la del modelo antes de contraer.
- **`confidence_level`**: se basa en `max(p, 1 − p)`.
  - *baja*: zona ambigua, fuera de distribución o advertencias.
  - *alta*: ≥ 0,90, el corte desde el cual la accuracy local en validación es ≥ 97 %. Nunca es "alta"
    si el puntaje OOD supera 0,756.
  - *media*: el resto.
  - Por la contracción, en textos tipo ISOT el sistema suele decir "media" aunque acierte (es
    conservador, ver informe §15.4).
- **`ood_score`**: 0 = muy parecido a las noticias de entrenamiento; 1 = muy distinto.
- **`top_terms`**: n-gramas de caracteres del texto con más peso hacia cada clase.
- **La probabilidad es parecido con el entrenamiento, no certeza sobre la verdad de los hechos.**

## 8. Tests

```bash
make test                        # o: .venv/bin/pytest -q        (Windows: python -m pytest -q)
make test-fast                   # sin el notebook (marcador "slow")
```

**153 tests.** Cubren:

- datos y splits sin solapamiento;
- limpieza v1/v2/v3 (idempotencia, contracciones, unicode);
- rasgos, modelos y métricas;
- contrato de `predict` y casos borde;
- invarianza del modelo final;
- atajos (las marcas y metadatos no entran al modelo);
- OOD y abstención;
- folds del panel y orden "selección antes de tocar el test";
- artefactos, demo y notebook.

Los tests que necesitan datos o modelos se **saltean** con un mensaje si todavía no se generaron.

## 9. Reproducir las cifras del informe

- Semilla única: `seed: 42` en `config.yaml`. Todos los parámetros (grillas, umbrales, candidatos del
  panel, perfil de limpieza) están en ese archivo.
- `make all` regenera todos los reportes y figuras. Las cifras del informe salen de `reports/*.json`
  y `reports/*.csv`.
- **Log de accesos al test**: `reports/test_access_log.jsonl`. Es append-only y tiene una línea por
  modelo y por conjunto. La iteración 2 registró 5 accesos, sin re-accesos. Si se vuelve a correr
  `evaluate` o `evaluate-v2`, el log marca la repetición como `reacceso: true`.
- Para regenerar solo figuras desde reportes ya existentes, sin tocar el test, cada módulo tiene su
  función de gráfico en `src/fakenews/plots.py`. `ablation_report()` (en `panel.py`) rehace la tabla y
  la figura de ablación.

## 10. Resultados reales

Fuente: `reports/comparison_it1_vs_it2.csv` y `reports/final_v2.json`.

| Conjunto | Métrica | Iteración 1 | Iteración 2 |
|---|---|---|---|
| ISOT test | accuracy | 0,989 | 0,990 |
| ISOT test | F1 falsa | 0,988 | 0,989 |
| Panel: cambio de subject | F1 macro | 0,739 | 0,737 |
| Panel: cambio temporal | F1 macro | 0,905 | 0,896 |
| Panel: WELFake-dev | F1 macro | 0,520 | 0,594 |
| **WELFake holdout** (trivial 0,583) | accuracy | 0,547 | **0,581** |
| WELFake holdout | ECE | 0,398 | **0,101** |
| LIAR test (no virgen; trivial 0,574) | accuracy | 0,473 | 0,514 |

**Limitaciones y riesgos éticos:**

- El modelo reconoce **estilo de fuente**, no veracidad. Fuera de su dataset no supera al trivial.
- Solo inglés, política de EE.UU. entre 2015 y 2018. Falla con sátira y con noticias nuevas.
- Un falso positivo puede desacreditar a un medio legítimo. Un falso negativo da un falso respaldo.
- **Nunca debe usarse como único criterio de moderación.**

Riesgos completos en el informe §12 y §15.8.

## 11. Estructura del repo y dónde cambiar cosas

```
config.yaml            ← TODOS los parámetros: semilla, grillas, umbral y zona ambigua (decision),
                         perfiles (cleaning), candidatos del panel (panel), OOD (ood_v2), modelo en uso (inference)
src/fakenews/
  cleaning.py          perfiles de limpieza v1 / v2 / v3 (misma función en entrenamiento e inferencia)
  data.py, corpus.py   descarga y carga (ISOT, LIAR, WELFake), textos por perfil
  splits.py, dedup.py, audit.py, shortcuts.py   splits, MinHash, auditoría de fuga y de atajos
  features.py, models.py, explain.py            TF-IDF, rasgos de estilo, modelos, términos influyentes
  train.py, evaluate.py                         iteración 1
  panel.py, invariance.py, ood.py, train_v2.py, evaluate_v2.py   iteración 2
  predict.py, demo_logic.py, cli.py             inferencia, demo, CLI
app/demo.py            demo Streamlit
examples/              textos de prueba (del dataset de validación)
docs/                  exploración, iteración 2, data card, informe, labs, campos del acta, guion
reports/               métricas, figuras, log de accesos al test
models/                metadata y parámetros OOD (versionados); .joblib (no versionados)
tests/                 pytest
```

- **Umbral y zona ambigua**: `config.yaml → decision` (`ambiguous_low`, `ambiguous_high`). Se aplican
  al reentrenar (`train-v2`).
- **Perfil de limpieza**: el modelo usa el perfil guardado en su artefacto. Para probar otro, agregar
  un candidato en `panel.candidates` y fijar `final_v2.candidate`.

## 12. Solución de problemas

| Problema | Solución |
|---|---|
| `Falta data/raw/isot_kaggle/Fake.csv` o `Falta data/processed/train.csv` | correr `make data` (o `quickstart`). Si la descarga falla, usar la alternativa manual de §4. |
| `No se encontró models/final_model...joblib` (CLI o demo) | correr `make quickstart` o `make train-v2`; la demo muestra el comando |
| `ModuleNotFoundError: fakenews` | el entorno no está activado o falta `pip install -e . --no-deps` |
| `Port 8501 is already in use` | `streamlit run app/demo.py --server.port 8502` |
| Errores de versión de librerías | usar Python 3.12 y reinstalar con `pip install -r requirements.txt` (versiones fijadas) |
| El panel se queda sin memoria | `python -m fakenews.cli panel --jobs 1` |
| Muchos tests "skipped" | es normal si todavía no se generaron datos o modelos |

## 13. Créditos, licencias y cita de los datasets

- **ISOT Fake News Dataset**: H. Ahmed, I. Traore y S. Saad (2017, 2018), University of Victoria.
  Mirror de Kaggle de C. Bisaillon (CC BY-NC-SA 4.0).
- **LIAR**: W. Y. Wang. *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News
  Detection.* ACL 2017. Solo para investigación.
- **WELFake**: P. K. Verma, P. Agrawal y R. Prodan. *WELFake dataset for fake news detection in text
  data*. Zenodo, 2021, doi:10.5281/zenodo.4561253. CC BY 4.0.
- Código: proyecto académico del equipo (UCA, 2026).
