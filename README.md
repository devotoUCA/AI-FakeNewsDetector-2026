# Detector de Fake News

Proyecto de la materia **Inteligencia Artificial (UCA, 2026)**. Docentes: Germán Lamartine, Felicitas Klix y Javier
Adolfo Ouret. Equipo: Tomás Cortina, Martín Coronel, Mateo Denti, Mateo Devoto y Fabricio Pompozzi.

## 1. Qué es y qué NO es

**Es** un clasificador de noticias **en inglés**, que usa **solo el texto** (título y/o cuerpo). Si el
texto se parece a sus datos de entrenamiento, devuelve una etiqueta, la **probabilidad de que sea falsa**,
un nivel de confianza y los términos que más pesaron. Si no se parece, **no emite veredicto**.

**No es** un verificador de hechos ni un producto para moderar contenido real.

> **Hallazgo principal.** El modelo aprende **patrones de redacción de las fuentes** del dataset, no la
> veracidad. Acierta el 99 % dentro de su dataset (ISOT), pero en un dataset donde ambas clases vienen
> de los mismos medios su capacidad de discriminar es la del azar. Detalle en el
> [informe final](docs/informe_resultados.md) y la [model card](docs/model_card.md).

## 2. Requisitos

| | |
|---|---|
| Python | 3.10 a 3.12. Probado con **3.12.13**. |
| Sistema operativo | **macOS (arm64): probado.** **Windows 11 (PowerShell, Python 3.12): probado** de punta a punta: instalación, tests, inicio rápido y demo. **Linux (x86_64): probados `setup` y los tests sin datos**; el inicio rápido y la demo, no. El CI (`.github/workflows/ci.yml`) corre los tests en Ubuntu y Windows. Detalle en [docs/03_cierre.md](docs/03_cierre.md). |
| Disco | ~0,9 GB para el entorno virtual. Inicio rápido: ~0,5 GB más. Pipeline completo: ~2,5 GB más. |
| Memoria | 8 GB alcanzan para el inicio rápido; 16 GB recomendados para los paneles (o `panel --jobs 1`). |
| Internet | Solo para instalar dependencias y descargar datos. Una vez entrenado, todo funciona **offline**. |

Tiempos medidos en un Apple M5 (10 núcleos, 16 GB):

| Paso | Tiempo |
|---|---|
| `setup` | ~20 s con caché de pip; más la primera vez |
| `quickstart` | ~7 min (medido: 6 min 47 s en un entorno limpio); ~11 min en una notebook con Windows |
| `data` | ~12 min (incluye descargar WELFake, 245 MB) |
| `train` (todas las iteraciones) | ~1 h 10 min (paneles ~40 min; modelo final ~12 min) |
| `evaluate` | ~8 min |
| `analyze` + `notebook` + `report` | ~4 min |

`data`, `train` y `evaluate` son sumas de pasos medidos por separado: `all` no se corrió de punta a punta
en la iteración 3, porque hacerlo implica volver a mirar los conjuntos reservados.

## 3. Instalación paso a paso

Hay dos caminos equivalentes: `make` (macOS/Linux) o `python -m fakenews.tasks` (todas las plataformas).
El Makefile es un envoltorio fino del segundo.

**macOS / Linux:**

```bash
git clone https://github.com/devotoUCA/AI-FakeNewsDetector-2026.git
cd AI-FakeNewsDetector-2026
make setup                       # o: PYTHONPATH=src python3.12 -m fakenews.tasks setup
.venv/bin/python -m fakenews.cli --help   # verificación: debe listar los comandos
```

**Windows (PowerShell; probado en Windows 11):**

```powershell
git clone https://github.com/devotoUCA/AI-FakeNewsDetector-2026.git
cd AI-FakeNewsDetector-2026
$env:PYTHONPATH="src"; py -3.12 -m fakenews.tasks setup
.\.venv\Scripts\python -m fakenews.cli --help
```

Después de `setup`, las tareas se corren así: `python -m fakenews.tasks <tarea>`, con cualquier Python
que tenga acceso a `src` (usan `.venv` automáticamente). En los ejemplos de abajo, en Windows cambiá
`.venv/bin/python` por `.\.venv\Scripts\python`.

## 4. Datos

Se descargan automáticamente.

| Dataset | Uso | Descarga automática | Alternativa manual | Licencia |
|---|---|---|---|---|
| ISOT / *Fake and Real News* (Kaggle) | entrenamiento | `cli download` (endpoint público, sin login) | https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset → `Fake.csv` y `True.csv` en `data/raw/isot_kaggle/` | **CC BY-NC-SA 4.0** según Kaggle. El ReadMe oficial del ISOT (University of Victoria) no declara licencia; solo pide citar dos trabajos de los autores (§13). |
| LIAR | externo | `cli download` | https://www.cs.ucsb.edu/~william/data/liar_dataset.zip → `data/raw/liar/` | **solo investigación** |
| WELFake | externo | `cli download-external` | https://zenodo.org/records/4561253 → `data/raw/welfake/` | **CC BY 4.0**. Ojo: su descripción está invertida respecto de su CSV. |
| FA-KES | holdout virgen | `cli download-external` | https://zenodo.org/records/2607278 → `data/raw/fakes/` | **CC BY 4.0** |

**Uso académico no comercial.** Los datos no se versionan. SHA-256 en `config.yaml`.

## 5. Inicio rápido (~7 minutos)

Descarga ISOT y LIAR, prepara los datos y entrena **solo el modelo final**. Reutiliza la compuerta de
dominio ya calibrada que viene en el repo (`models/domain_gate_v3.json.gz`):

```bash
make quickstart                  # o: python -m fakenews.tasks quickstart
.venv/bin/python -m fakenews.cli predict - < examples/01_agencia_real_val.txt
```

**Usar un modelo ya entrenado** (si el equipo lo comparte por fuera de git): copiar
`final_model_v3.joblib` y `retrieval_v3.joblib` en `models/`. Pesan ~40 MB y no se versionan.

## 6. Pipeline completo

```bash
make setup && make all           # o: python -m fakenews.tasks all
```

| Tarea (`make` o `tasks`) | Qué hace | Dónde mirar |
|---|---|---|
| `data` | descarga, limpia, deduplica, audita; prepara WELFake y FA-KES | `reports/splits_manifest.json`, `welfake_manifest_v2.json`, `fakes_manifest_v3.json` |
| `train` | iteraciones 1, 2 y 3: modelos, auditorías, paneles, modelo final con compuerta | `reports/panel_v3.csv`, `format_shortcut_discovery_v3.csv`, `domain_gate_v3.json`, `invariance_v3.json` |
| `evaluate` | evaluaciones finales (registradas en el log) | `reports/final_v3.json`, `error_analysis_v3.csv`, `acta_comparison_v3.csv` |
| `analyze` | clustering y algoritmo genético (exploratorios) | `reports/clustering.json` |
| `review` | cuenta los errores revisados y firmados por el equipo (no carga modelo ni datos) | `reports/error_analysis_v3.csv`, [docs/03_cierre.md](docs/03_cierre.md) |
| `report` | actualiza el criterio de revisión, regenera las tablas de los documentos y verifica el log | `docs/`, `README.md` |
| `notebook` | ejecuta el EDA | `notebooks/` |
| `test` / `test-fast` | tests | — |
| `demo` | demo web local | http://localhost:8501 |

Cada paso individual está en `python -m fakenews.cli --help`. **Atención:** volver a correr `evaluate`
deja nuevos accesos en el log, marcados como `reacceso`.

## 7. Cómo probar el modelo

### Demo

```bash
make demo                        # o: python -m fakenews.tasks demo
```

Abrir http://localhost:8501, pegar un texto en inglés (o elegir un ejemplo) y apretar **Analizar**.

- **Dentro de dominio:** aviso verde, etiqueta, probabilidad, confianza, zona ambigua, términos
  influyentes y noticias de entrenamiento parecidas.
- **Fuera de dominio:** "este texto no se parece a las noticias con las que se entrenó el modelo; no se
  emite veredicto", con el valor de referencia detrás de un desplegable marcado "NO confiable".

### CLI

```bash
.venv/bin/python -m fakenews.cli predict "texto en inglés..."
.venv/bin/python -m fakenews.cli predict - < archivo.txt
.venv/bin/python -m fakenews.cli predict - --similar < archivo.txt
```

En **PowerShell** no existe la redirección `<`. Para leer un archivo:

```powershell
Get-Content -Raw -Encoding UTF8 examples\01_agencia_real_val.txt | .\.venv\Scripts\python -m fakenews.cli predict -
```

**Dentro de dominio** (salida real recortada, `examples/01_agencia_real_val.txt`):

```json
{
  "label": "probablemente real",
  "p_false": 0.001,
  "p_false_display": "< 1 %",
  "confidence_level": "alta",
  "in_domain": true,
  "domain_score": 0.5492,
  "warnings": []
}
```

**Fuera de dominio** (salida real recortada, `examples/03_afirmacion_liar_valid.txt`):

```json
{
  "label": "incierta, requiere revisión humana",
  "p_false": null,
  "p_false_display": "sin veredicto",
  "confidence_level": "baja",
  "in_domain": false,
  "domain_score": 0.0178,
  "p_false_reference": 0.7735,
  "warnings": ["fuera_de_dominio: no se parece a las noticias de entrenamiento (puntaje de dominio 0.02 < 0.17). No se emite veredicto."]
}
```

### Casos de prueba sugeridos (resultados reales)

Los textos de `examples/` salen de ISOT validación y de LIAR-valid (ver `examples/README.md`).

| Caso | Entrada | Resultado |
|---|---|---|
| Nota tipo agencia (real en el dataset) | `examples/01_agencia_real_val.txt` | *probablemente real*, < 1 %, confianza alta, dentro de dominio |
| Tono sensacionalista (falsa en el dataset) | `examples/02_sensacionalista_falsa_val.txt` | *probablemente falsa*, > 99 %, confianza alta, dentro de dominio |
| Afirmación tipo LIAR | `examples/03_afirmacion_liar_valid.txt` | sin veredicto (fuera de dominio) |
| Texto muy corto | `"Breaking news today"` | sin veredicto, "Texto muy corto" |
| Texto en español | `"El gobierno anunció hoy un nuevo plan económico que incluye una reducción de impuestos para las pequeñas empresas y un aumento del gasto en infraestructura."` | sin veredicto, "no parece estar en inglés" |
| Solo URL | `"https://example.com/noticia"` | sin veredicto, "solo contiene enlaces" |
| Vacío | `""` | sin veredicto, "El texto está vacío." |

### Cómo interpretar la salida

- **`in_domain`**: si el texto se parece a ISOT. **No significa que la respuesta sea correcta**, y la
  compuerta deja pasar parte de los estilos que no conoce.
- **`label`**: fuera de dominio, *incierta, requiere revisión humana*, con la advertencia
  `fuera_de_dominio`. Dentro: *incierta* si la probabilidad cae en la **zona ambigua 0,4–0,6**; si no,
  compara con el umbral 0,50.
- **`p_false`**: probabilidad calibrada, solo dentro de dominio (nunca 0 ni 1 literales).
  `p_false_display` muestra "< 1 %" o "> 99 %" en los extremos. `p_false_reference` conserva el valor
  fuera de dominio: **no es confiable**.
- **`confidence_level`**: según `max(p, 1 − p)`. *baja*: zona ambigua o fuera de dominio. *alta*: ≥ 0,90.
  *media*: el resto.
- **`top_terms`**: n-gramas de caracteres del texto con más peso hacia cada clase.
- **La probabilidad es parecido con el entrenamiento, no certeza sobre la verdad de los hechos.**

## 8. Tests

```bash
make test                        # o: python -m fakenews.tasks test
make test-fast                   # sin el notebook
```

**240 tests.** Sin datos ni modelos, que es lo que corre el CI (`-m "not slow"`): 180 pasan, 58 se
saltean y 1 es `xfail` (medido en Linux). Con todos los artefactos se midieron 230 que pasan y 1 `xfail`
en macOS, antes de sumar los 9 tests de la revisión de errores; esos 9 no necesitan artefactos y pasan,
así que el total esperado es 239 y 1 `xfail` (195, 44 salteados y 1 `xfail` tras el inicio rápido). Cubren datos y splits, limpieza (v1 a v3), modelos y métricas, contrato de `predict`,
compuerta de dominio, invarianza con IC, atajos, log de accesos, consistencia entre documentos y
reportes, revisión de errores, runner, demo y notebook. Un test figura como `xfailed` **a propósito**: documenta que el modelo
final no cumple la invarianza en `oraciones_pegadas`. Los tests que necesitan datos o modelos se saltean
si todavía no se generaron.

## 9. Reproducir las cifras del informe

- Semilla única: `seed: 42` en `config.yaml`, que tiene todos los parámetros.
- **Las tablas del informe, de este README y de `docs/campos_completar.md` se generan desde `reports/`**:
  `python -m fakenews.cli report`. Si una cifra no coincide, el test de consistencia falla.
- **Log de accesos:** `reports/test_access_log.jsonl`, append-only. Verificación:
  `.venv/bin/python -m fakenews.access_log`.
- Figuras: `reports/figures/`, generadas por los mismos comandos.

## 10. Resultados reales

Modelo final, por conjunto. ISOT test y WELFake holdout ya se habían mirado antes; **FA-KES es la única
cifra virgen**.

<!-- BEGIN GENERATED:readme -->
| Conjunto | Accuracy (forzando respuesta) | Trivial | Fuera de dominio | Accuracy sobre lo respondido | Trivial sobre lo mismo |
|---|---|---|---|---|---|
| ISOT test | 0,990 | 0,547 | 2,9 % | 0,990 | 0,561 |
| WELFake holdout | 0,581 | 0,583 | 70,3 % | 0,640 | 0,502 |
| LIAR test (no virgen) | 0,514 | 0,574 | 92,1 % | 0,508 | 0,460 |
| FA-KES (virgen) | 0,531 | 0,530 | 40,8 % | 0,538 | 0,540 |
<!-- END GENERATED:readme -->

Tres iteraciones:

<!-- BEGIN GENERATED:iteraciones -->
| Vista | It. 1 | It. 2 | It. 3 | Trivial |
|---|---|---|---|---|
| ISOT val, F1 macro | 0,990 | 0,989 | 0,989 | — |
| Cambio de subject, F1 macro | 0,739 | 0,737 | 0,737 | — |
| Cambio temporal, F1 macro | 0,905 | 0,896 | 0,896 | — |
| WELFake dev, F1 macro | 0,520 | 0,594 | 0,594 | — |
| LIAR valid, F1 macro | 0,504 | 0,567 | 0,567 | — |
| Puntaje robusto | 0,721 | 0,742 | 0,742 | — |
| ISOT test, accuracy | 0,989 | 0,990 | 0,990 | 0,547 |
| ISOT test, F1 falsa | 0,988 | 0,989 | 0,989 | — |
| WELFake holdout, accuracy | 0,547 | 0,581 | 0,581 | 0,583 |
| WELFake holdout, F1 macro | 0,525 | 0,575 | 0,575 | — |
| LIAR test, accuracy | 0,473 | 0,514 | 0,514 | 0,574 |
| FA-KES, accuracy | 0,511 | 0,531 | 0,531 | 0,530 |
| FA-KES, F1 macro | 0,417 | 0,398 | 0,398 | — |
<!-- END GENERATED:iteraciones -->

**Limitaciones y riesgos éticos:**

- Reconoce **estilo de fuente**, no veracidad. En FA-KES la discriminación es la del azar.
- **No cumple el criterio de invarianza** en `oraciones_pegadas`: depende de un artefacto de formato.
- La compuerta solo reconoce bien lo que vio como negativo.
- Solo inglés, política de EE.UU. entre 2015 y 2018. Falla con sátira y noticias nuevas.
- Un falso positivo puede desacreditar a un medio legítimo; un falso negativo da falso respaldo.
- **Nunca debe usarse como único criterio de moderación.**

## 11. Estructura del repo y dónde cambiar cosas

```
config.yaml            ← todos los parámetros
src/fakenews/
  cleaning.py          perfiles de limpieza v1 / v2 / v2_1 / v3
  data.py, corpus.py   descarga y carga (ISOT, LIAR, WELFake, FA-KES)
  splits.py, dedup.py, audit.py, shortcuts.py, format_shortcuts.py   splits y auditorías
  features.py, models.py, explain.py, embeddings.py                  rasgos y modelos
  train.py, evaluate.py                          iteración 1
  panel.py, invariance.py, ood.py, train_v2.py, evaluate_v2.py       iteración 2
  domain.py, train_v3.py, evaluate_v3.py         iteración 3 (modelo final)
  predict.py, demo_logic.py, cli.py, tasks.py    inferencia, demo, CLI, runner
  access_log.py, make_report.py                  log de accesos y tablas de los documentos
  review.py                                      conteo de errores revisados y firmados por el equipo
app/demo.py            demo Streamlit
examples/              textos de prueba (de validación)
docs/                  informe, model card, data card, guion, bitácora, decisiones por iteración, cierre
reports/               métricas, figuras, log de accesos
models/                metadata y compuerta (versionados); .joblib (no versionados)
```

- **Umbral y zona ambigua:** `config.yaml → decision`.
- **Retención de la compuerta:** `config.yaml → domain_gate.min_in_domain_retention`.
- **Modelo final:** `config.yaml → final_v3.candidate`.
- Los cambios se aplican al reentrenar (`python -m fakenews.cli train-v3`).

## 12. Solución de problemas

| Problema | Solución |
|---|---|
| `Falta data/raw/...` o `Falta data/processed/...` | correr `quickstart` o `data`; si la descarga falla, usar la alternativa manual de §4 |
| `No se encontró models/...joblib` | correr `quickstart`; la demo muestra el comando |
| `ModuleNotFoundError: fakenews` | falta `setup`, o falta `PYTHONPATH=src` para llamar a `fakenews.tasks` antes de instalar |
| `Port 8501 is already in use` | `.venv/bin/streamlit run app/demo.py --server.port 8502` |
| Errores de versión de librerías | usar Python 3.12 y reinstalar con `pip install -r requirements.txt` |
| Un panel se queda sin memoria | `python -m fakenews.cli panel-v3 --jobs 1` |
| PowerShell no deja activar scripts | no hace falta activar: usar `.\.venv\Scripts\python` directamente |
| `py -3.12` dice "No suitable Python runtime found" | instalar Python 3.12 (`winget install -e --id Python.Python.3.12` o desde python.org) y reabrir PowerShell |
| Windows: al cerrar la demo aparece `ConnectionResetError: [WinError 10054]` y queda en "Stopping..." | es un aviso de Python en Windows al cortarse la conexión del navegador; no afecta los resultados. Apretar Ctrl + C otra vez o cerrar la ventana |
| Muchos tests "skipped" | normal si todavía no se generaron datos o modelos |

## 13. Créditos, licencias y cita de los datasets

- **ISOT Fake News Dataset** (University of Victoria). Mirror de Kaggle de C. Bisaillon (CC BY-NC-SA 4.0).
  Citas que pide el ReadMe oficial del dataset:
  - H. Ahmed, I. Traore y S. Saad. *Detecting opinion spams and fake news using text classification*.
    Journal of Security and Privacy, vol. 1, n.º 1, Wiley, enero/febrero 2018.
  - H. Ahmed, I. Traore y S. Saad. *Detection of Online Fake News Using N-Gram Analysis and Machine
    Learning Techniques*. En: I. Traore, I. Woungang y A. Awad (eds.), *Intelligent, Secure, and
    Dependable Systems in Distributed and Cloud Environments* (ISDDC 2017), Lecture Notes in Computer
    Science, vol. 10618, Springer, Cham, 2017, pp. 127–138.
- **LIAR**: W. Y. Wang. *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News
  Detection.* ACL 2017. Solo para investigación.
- **WELFake**: P. K. Verma, P. Agrawal y R. Prodan. *WELFake dataset for fake news detection in text
  data*. Zenodo, 2021, doi:10.5281/zenodo.4561253. CC BY 4.0.
- **FA-KES**: F. K. Abu Salem, R. Al Feel, S. Elbassuoni, M. Jaber y M. Farah. *FA-KES: A Fake News
  Dataset around the Syrian War*. Zenodo, 2019, doi:10.5281/zenodo.2607278. CC BY 4.0.
- Embeddings opcionales: `sentence-transformers/all-MiniLM-L6-v2` (`requirements-extra.txt`).
- Código: proyecto académico del equipo (UCA, 2026).
