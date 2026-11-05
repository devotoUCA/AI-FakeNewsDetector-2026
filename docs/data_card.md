# Data card

Todos los números salen de `reports/data_audit.json`, `reports/leakage_audit.json` y
`reports/splits_manifest.json` (corrida del 04/11/2026, semilla 42).

## 1. Dataset principal: ISOT / *Fake and Real News Dataset* (Kaggle)

| Campo | Valor |
|---|---|
| Fuente | Kaggle, `clmentbisaillon/fake-and-real-news-dataset` (versión 1, actualizada 19/04/2024). Fuente declarada: **ISOT Fake News Dataset**, University of Victoria (Ahmed, Traore y Saad, 2017–2018). |
| Descarga | Endpoint público de Kaggle, sin login (`make data`). SHA-256 de `Fake.csv` y `True.csv` en `config.yaml`. |
| Licencia | **CC BY-NC-SA 4.0**, según declara Kaggle. Uso académico no comercial con atribución: compatible con el proyecto. La licencia del ISOT original no se verificó por separado (ver "Pendientes"). |
| Idioma | Inglés |
| Archivos | `Fake.csv` (23 481 filas), `True.csv` (21 417 filas) |
| Columnas | `title`, `text`, `subject`, `date` |
| Clases | **Falsa (1, positiva)**: artículos de sitios que PolitiFact y Wikipedia señalaron como poco confiables. **Real (0)**: artículos de la agencia **Reuters**. |
| Qué significa la etiqueta | La etiqueta es *de la fuente*, no de cada artículo: "real" quiere decir "publicado por Reuters" y "falsa" quiere decir "publicado por un sitio señalado como poco confiable". Ninguna noticia se verificó una por una. |
| Período | Reales: 13/01/2016 a 31/12/2017. Falsas: 31/03/2015 a 19/02/2018. |
| Temas | Política de EE.UU. y noticias internacionales. `subject` de las reales: `politicsNews` (11 272) y `worldnews` (10 145). De las falsas: `News` (9050), `politics` (6841), `left-news` (4459), `Government News` (1570), `US_News` (783), `Middle-east` (778). |

### Calidad (dataset crudo)

| Indicador | Falsas | Reales |
|---|---|---|
| Filas | 23 481 | 21 417 |
| Cuerpo vacío | 630 | 1 |
| Cuerpo con < 20 palabras | 1105 | 1 |
| Cuerpo con solo enlaces | 84 | 0 |
| Fechas que no se pueden leer (son URLs en la columna `date`) | 10 | 0 |
| Cuerpos con > 5000 palabras | 28 | 1 |
| Duplicados exactos del cuerpo | 6026 | 225 |
| Palabras por cuerpo (mediana / p99 / máximo) | 375 / 1975 / 8375 | 367 / 1179 / 5292 |
| Títulos nulos | 0 | 0 |
| Textos idénticos entre clases (sin contar vacíos) | 0 | 0 |

### Fuga de información (medida en el texto crudo)

La "accuracy de la regla sola" es la que logra una regla que mira **únicamente** esa marca.

| Marca | % falsas | % reales | Accuracy de la regla sola |
|---|---|---|---|
| dateline `CIUDAD (Reuters) -` | 0,0 | 99,2 | **0,996** |
| palabra "Reuters" | 1,3 | 99,8 | 0,993 |
| título con palabra en MAYÚSCULAS (≥ 4 letras) | 79,6 | 2,5 | 0,881 |
| apóstrofo convertido en espacio (`don t`) | 65,4 | 8,9 | 0,777 |
| comillas/apóstrofos curvos (`’ “ ”`) | 0,0 | 50,5 | 0,764 |
| "Featured image" | 34,8 | 0,0 | 0,659 |
| `@usuario` | 26,1 | 1,3 | 0,607 |
| firma de tuit `(@x) Mes D, AAAA` | 23,2 | 0,0 | 0,598 |
| `(VIDEO)` en el título | 21,5 | 0,0 | 0,590 |
| `Via:` / `Read more:` | 20,9 | 0,0 | 0,586 |

Metadatos solos (se entrena con train y se mide en validación): **`subject` → accuracy 1,000**;
**formato del string de fecha** (espacio final, mes abreviado) → **1,000**; año/mes/día → 0,720.

### Preparación (filas perdidas en cada paso)

| Paso | Reales | Falsas | Total |
|---|---|---|---|
| Carga | 21 417 | 23 481 | 44 898 |
| − textos con < 20 palabras tras limpiar | −3 | −726 | −729 |
| − duplicados exactos (texto limpio normalizado) | −490 | −5381 | −5871 |
| − casi-duplicados (MinHash, Jaccard ≥ 0,8; 103 pares, 1 grupo con etiquetas en conflicto descartado entero) | −40 | −59 | −99 |
| **Dataset final** | **20 884** | **17 315** | **38 199** |

Tras deduplicar, la clase mayoritaria pasa a ser **real (54,7 %)**.

| Split | Reales | Falsas | Total |
|---|---|---|---|
| train (70 %) | 14 619 | 12 120 | 26 739 |
| val (15 %) | 3133 | 2597 | 5730 |
| test (15 %) | 3132 | 2598 | 5730 |

Solapamiento entre splits: 0 ids, 0 duplicados exactos y 0 casi-duplicados para todos los pares
(`reports/splits_manifest.json`, que también guarda el SHA-256 de cada archivo y de su lista de ids).

### Sesgos y limitaciones del dataset

- **Sesgo de fuente**: el 100 % de las reales son de Reuters. El modelo puede aprender "estilo agencia
  vs. estilo blog" en lugar de "verdadero vs. falso".
- **Sesgo de período y tema**: política de EE.UU. entre 2015 y 2018 (elección de 2016, primer año de
  Trump). Nada indica que funcione con otros temas, otros años u otros países.
- **Etiqueta por fuente**: hay artículos de Reuters que reproducen textualmente tuits sensacionalistas
  (etiquetados "real") y artículos de sitios poco confiables que copian notas de agencia (etiquetados "falsa").
- **Sátira y opinión** quedan dentro de "falsa" sin distinción.
- Solo inglés: los resultados no se pueden extrapolar al español.

## 2. Dataset externo: LIAR (solo evaluación, nunca entrenamiento)

| Campo | Valor |
|---|---|
| Fuente | W. Y. Wang, *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection*, ACL 2017. https://www.cs.ucsb.edu/~william/data/liar_dataset.zip |
| Licencia | Su README dice: "You are allowed to use this dataset for research purposes only" y "The original sources retain the copyright of the data". No declara una licencia estándar. |
| Contenido | 12 836 afirmaciones cortas de políticos de EE.UU. verificadas por PolitiFact, con 6 etiquetas |
| Conteos | half-true 2638, false 2511, mostly-true 2466, barely-true 2108, true 2063, pants-fire 1050 |
| Binarización | **falsa** = `false` + `pants-fire`; **real** = `true` + `mostly-true`; **se excluyen** `half-true` y `barely-true` porque no se pueden asignar a una clase sin forzar la etiqueta. Quedan **8090** afirmaciones (44,0 % falsas). |
| Diferencia de dominio | Son afirmaciones de unas 17 palabras (mediana), no artículos de unas 384. La etiqueta es de veracidad verificada, no de fuente. |

## 3. Dataset externo de la iteración 2: WELFake (solo evaluación)

| Campo | Valor |
|---|---|
| Fuente | P. K. Verma, P. Agrawal y R. Prodan, *WELFake dataset for fake news detection in text data*, Zenodo, registro 4561253 (2021). https://zenodo.org/records/4561253 |
| Licencia | **CC BY 4.0**, declarada en Zenodo |
| Contenido | 72 134 artículos (título y texto) que combinan cuatro datasets: Kaggle, McIntire, **Reuters (= ISOT)** y BuzzFeed Political |
| **Dirección de las etiquetas** | La descripción de Zenodo dice "0 = fake, 1 = real", **pero el CSV usa 1 = falsa y 0 = real**. Se verificó empíricamente con las 44 284 filas que coinciden con ISOT: el acuerdo con "1 = falsa" es del 99,97 %. Coincide con nuestra convención y no se invierte nada (`corpus.verify_welfake_label_direction` falla si esto cambia). |
| Solapamiento con ISOT | **61 %** (44 284 filas). Se quitaron todas las filas idénticas a cualquier fila de ISOT (44 198), los casi-duplicados de ISOT (MinHash, Jaccard ≥ 0,8: 42) y las coincidencias con LIAR (0). |
| Limpieza | Textos < 20 palabras (1209); duplicados exactos internos (2602) y casi-duplicados internos (833) |
| Resultado | **23 250 artículos** que no están en ISOT (41,7 % falsas); solo el 0,1 % tiene dateline de Reuters |
| Uso | 50/50 estratificado con semilla 42: **dev** (11 625; 4850 falsas) para decisiones y **holdout** (11 625; 4849 falsas) para la evaluación final única de la iteración 2 |
| Limitaciones | Las etiquetas también son por fuente, según cada dataset de origen. Tras deduplicar quedan sobre todo los componentes Kaggle y McIntire, que tienen sus propios sesgos de fuente y de período (alrededor de 2016). |

Manifiesto: `reports/welfake_manifest_v2.json` (SHA-256 del CSV, pasos y hashes de ids de cada split).

### Porciones de LIAR

En la iteración 1 se evaluaron las tres porciones juntas, así que **ninguna es virgen**. En la iteración 2:

- `train` se usa para entrenar en los experimentos D1.
- `valid` se usa como desarrollo externo.
- `test` se reporta marcado como "no virgen".

## Pendientes (para el equipo)

- Confirmar si el ISOT original de la University of Victoria impone condiciones además de la licencia
  CC BY-NC-SA 4.0 que declara el mirror de Kaggle.
- Confirmar las condiciones de uso de WELFake: CC BY 4.0, con cita de Verma, Agrawal y Prodan (2021).
- Revisar a mano una muestra de etiquetas dudosas (acta, riesgo "etiquetas ambiguas"):
  `reports/error_analysis.csv` incluye casos de este tipo.
