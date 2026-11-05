# Bitácora de iteraciones (texto histórico)

> **Apéndice del informe final.** Este es el informe tal como quedó al cerrar la iteración 2, sin
> cambios, para conservar la historia. Sus conclusiones fueron **corregidas por la iteración 3**:
>
> - "La generalización mejoró" (§15.5): la mejora en WELFake fue sobre todo un corrimiento de sesgo
>   ligado al atajo de oraciones pegadas, no mejor discriminación.
> - El atajo de oraciones pegadas estaba subestimado (§15.3): el 1,25 % se midió en un solo sentido;
>   la prueba rigurosa da 7,65 %.
> - La contracción OOD (§15.4) se quitó.
>
> El documento vigente es `docs/informe_resultados.md`.

---

# Informe de resultados — Detector de Fake News

**Materia:** Inteligencia Artificial (UCA, 2026) · **Docentes:** Germán Lamartine, Javier Adolfo Ouret
**Equipo:** Tomás Cortina, Martín Coronel, Mateo Denti, Mateo Devoto, Fabricio Pompozzi

Todos los números se leen de los archivos de `reports/` indicados en cada sección (semilla 42, corrida
del 04/11/2026). El conjunto de prueba se evaluó **una sola vez por modelo**, después de congelar todas
las decisiones en validación (`reports/test_access_log.jsonl`: 11 accesos, uno por modelo, sin
re-accesos).

## 1. Resumen

> **Iteración 2 (ver §15).**
> - Modelo productivo nuevo: LR sobre n-gramas de caracteres con limpieza v2, elegido por un panel
>   de robustez.
> - ISOT test: 0,990.
> - WELFake-holdout (externo nuevo): accuracy 0,581 frente a 0,547 de la iteración 1. Sigue
>   **por debajo del trivial** (0,583).
> - ECE externa: 0,10 frente a 0,40.
> - Lo que sigue en este resumen describe la iteración 1.


- Se entrenó un clasificador de noticias en inglés (Real = 0, Falsa = 1) sobre el dataset ISOT/Kaggle,
  tras quitar las marcas de fuente y de formato y deduplicar antes de dividir.
- El **modelo productivo** es un **SVM lineal sobre TF-IDF (1–2 gramas), calibrado con regresión isotónica**.
  En test logra **accuracy 0,989**, **F1 falsa 0,988** y **F1 real 0,990**. El baseline (regresión
  logística TF-IDF) logra 0,986 / 0,985 / 0,988. La diferencia es chica pero significativa
  (McNemar p = 0,030).
- **Esos números se tratan con sospecha**, como pide el acta (accuracy > 98 %). La validación externa
  en LIAR, sin reentrenar, da **accuracy 0,487**, por debajo del clasificador trivial (0,560). El
  modelo reconoce el *estilo de las fuentes* de ISOT (Reuters frente a blogs), no la veracidad.
- Conclusión honesta: el sistema cumple los criterios numéricos del acta *dentro de ISOT*, pero **no
  es evidencia de que detecte noticias falsas en general**. Sirve como herramienta didáctica y de
  priorización, siempre con revisión humana.

## 2. Dataset

Detalle completo en `docs/data_card.md`.

- ISOT / *Fake and Real News Dataset* (Kaggle), licencia CC BY-NC-SA 4.0: 44 898 artículos (23 481
  falsos, 21 417 reales). Las reales son todas de Reuters; las falsas vienen de sitios señalados como
  poco confiables.
- Externo: LIAR (W. Y. Wang, 2017, "research purposes only"): 8090 afirmaciones tras binarizar.

## 3. Preparación y fugas encontradas

Fuente: `reports/leakage_audit.json`, `reports/splits_manifest.json`.

**Fugas medidas**, con la accuracy de una regla que mira solo esa marca:

- Dateline `(Reuters) -`: 0,996.
- Palabra "Reuters": 0,993.
- `subject`: 1,000.
- Formato del string de fecha: 1,000.
- Codificación tipográfica (comillas curvas solo en reales; apóstrofo convertido en espacio en el
  65 % de las falsas): 0,76–0,78. Este hallazgo **no figuraba en las hipótesis iniciales**.
- Créditos de imagen, embeds de tuits, URLs, `Via:`, promos de sitios y `(VIDEO)` en títulos:
  aparecen solo en falsas.

**Mitigación.**

- `subject` y `date` nunca entran al modelo.
- Una única función de limpieza (`src/fakenews/cleaning.py`), idéntica en entrenamiento e inferencia,
  con 27 reglas. Cada una se justifica con conteos. Tras limpiar, todas las marcas de la tabla quedan
  en 0 % en ambas clases.
- Se filtran los textos con < 20 palabras (729, casi todos falsas reducidas al título).
- Se eliminan duplicados exactos (5871) y casi-duplicados por MinHash (99) **antes** de dividir.
- Splits 70/15/15 estratificados, con semilla 42: 26 739 / 5730 / 5730. No hay solapamiento exacto ni
  casi-duplicado entre ningún par de splits.

Durante el desarrollo, la inspección de los términos más influyentes destapó dos marcas más que se
agregaron a la limpieza **antes** de evaluar en test:

- La plantilla de Reuters para tuits (`[0804 EST] Source link:`).
- Las instrucciones de video embebido de los blogs (`Watch the video below:`).

## 4. Modelos y comparación

Todos se entrenan solo con train. Los hiperparámetros se eligen por F1 de falsa en validación
(`reports/val_results.json`, `reports/baseline_logreg_grid.csv`).

| Modelo | Configuración elegida en validación |
|---|---|
| Trivial (`DummyClassifier`, prior) | predice la clase mayoritaria (real); probabilidad = proporción de falsas en train = 0,453 |
| **Baseline:** TF-IDF + regresión logística | C = 100, unigramas (40 060 términos); F1 falsa train 0,9995 / val 0,9851 (brecha 0,014). Hay una meseta desde C = 30 (0,9846). |
| Naive Bayes multinomial | α = 0,01, 1–2 gramas |
| SVM lineal | C = 3, 1–2 gramas (100 000 términos); F1 val 0,9894 |
| Árbol sobre rasgos de estilo | profundidad 12 |
| Bosque sobre rasgos de estilo | 200 árboles, profundidad 12 |
| Ensamble (stacking) | meta-LR sobre las probabilidades fuera de muestra de LR-texto y bosque-estilo; pesos 8,05 (texto) y 3,88 (estilo) |
| kNN coseno | k = 15 (grilla del acta: 3, 5, 7, 11, 15) |
| Red pequeña (MLP) | TF-IDF 20 000 → 32 ReLU → sigmoide; mejor época 24 de 30 por log-loss en validación |

### Resultados en test (una sola evaluación)

Fuente: `reports/test_comparison.csv` y `reports/test_results.json`. IC 95 % por bootstrap con 1000
remuestreos.

| Modelo | Accuracy | Prec. falsa | Rec. falsa | F1 falsa [IC 95 %] | Prec. real | Rec. real | F1 real | Brier | Log-loss |
|---|---|---|---|---|---|---|---|---|---|
| Trivial | 0,547 | 0,000 | 0,000 | 0,000 | 0,547 | 1,000 | 0,707 | 0,248 | 0,689 |
| LR TF-IDF (baseline) | 0,986 | 0,988 | 0,982 | 0,985 [0,981–0,988] | 0,985 | 0,990 | 0,988 | 0,0104 | 0,039 |
| Naive Bayes | 0,947 | 0,941 | 0,942 | 0,941 [0,935–0,948] | 0,952 | 0,951 | 0,951 | 0,0408 | 0,142 |
| **SVM lineal (productivo)** | **0,989** | 0,987 | 0,990 | **0,988 [0,985–0,991]** | 0,991 | 0,989 | **0,990** | **0,0082** | **0,032** |
| Árbol (estilo) | 0,900 | 0,889 | 0,890 | 0,890 [0,881–0,898] | 0,909 | 0,908 | 0,908 | 0,0743 | 0,665 |
| Bosque (estilo) | 0,927 | 0,924 | 0,913 | 0,919 [0,911–0,926] | 0,929 | 0,938 | 0,933 | 0,0555 | 0,194 |
| Ensamble | 0,987 | 0,987 | 0,985 | 0,986 [0,983–0,989] | 0,988 | 0,990 | 0,989 | 0,0096 | 0,039 |
| kNN coseno | 0,900 | 0,960 | 0,813 | 0,880 [0,871–0,891] | 0,863 | 0,972 | 0,914 | 0,0744 | 0,387 |
| Red pequeña (MLP) | 0,986 | 0,987 | 0,983 | 0,985 [0,982–0,988] | 0,986 | 0,989 | 0,988 | 0,0106 | 0,043 |

La fila "LR calibrada" se omite porque coincide con el baseline: la calibración elegida para la LR
fue "ninguna" (§6).

Figura: `reports/figures/f1_by_model_test.png`.

### Diferencias explicadas (McNemar exacto sobre test)

| A vs B | Solo A acierta | Solo B acierta | p |
|---|---|---|---|
| LR baseline vs **SVM** | 19 | 36 | **0,030** |
| LR baseline vs ensamble | 9 | 15 | 0,31 |
| LR baseline vs MLP | 31 | 31 | 1,00 |
| LR baseline vs bosque de estilo | 386 | 45 | < 10⁻⁶⁸ |

- **SVM frente a LR.** Ambos son lineales sobre TF-IDF. El SVM usa bigramas (100 000 términos) y la
  pérdida *hinge*, que se concentra en los casos cerca de la frontera. El resultado es menos falsos
  negativos (27 contra 48) a cambio de unos pocos falsos positivos más (34 contra 30). La mejora es
  de unos 0,3 puntos de accuracy: estadísticamente significativa, pero chica en términos prácticos.
- **Red pequeña.** Ganancia de F1 frente a la LR de +0,0008 en validación y +0,00003 en test. La
  regla del acta (adoptarla solo si mejora de forma consistente en validación y test, con un umbral
  de +0,005) dice **no adoptarla**. Se conserva el modelo lineal por simplicidad e interpretabilidad.
- **kNN.** k = 15 fue el mejor de la grilla (F1 val de 0,865 con k = 3 a 0,874 con k = 15). El
  16,4 % de los vecinos de un texto de validación tienen la otra etiqueta: los vecinos comparten tema
  pero no siempre veracidad (acta, formulario 9).

## 5. Matriz de confusión (test)

| | Pred. real | Pred. falsa |
|---|---|---|
| **Real** | 3098 (98,9 %) | 34 (1,1 %) — FP |
| **Falsa** | 27 (1,0 %) — FN | 2571 (99,0 %) |

Figuras: `reports/figures/confusion_test_productive.png` y `confusion_test_baseline.png`. Para el
baseline: TN 3102, FP 30, FN 48, TP 2550.

## 6. Probabilidad, calibración, umbral y nivel de confianza

Fuente: `reports/calibration_val.json`, `reports/decision_val.json` y `reports/test_results.json`.

### Calibración

Se usa validación cruzada de 5 folds dentro de train y se elige el método por Brier en validación.

| Modelo | Sin calibrar (Brier / ECE) | Sigmoide | Isotónica | Elegido |
|---|---|---|---|---|
| LR | 0,01055 / 0,0032 | 0,01098 / 0,0071 | 0,01092 / 0,0060 | ninguna |
| Naive Bayes | 0,04178 / 0,0290 | 0,04373 / 0,0093 | 0,03893 / 0,0087 | isotónica |
| SVM | (no tiene probabilidad propia) | 0,00784 / 0,0064 | 0,00782 / 0,0053 | isotónica |

La LR ya estaba bien calibrada, así que calibrarla empeoraba el Brier. En test, el productivo tiene
Brier 0,0082 y ECE 0,0041, contra 0,248 del baseline de probabilidad (predecir siempre 0,453).

Figuras: `reliability_logreg_val.png`, `reliability_productive_test.png`.

### Umbral

Criterio: máximo F1 de falsa con una tasa de falsas alarmas ≤ 10 % en validación; si hay empate, se
elige el umbral más cercano a 0,5. Resultado: **0,50** (F1 val 0,989, FPR 0,99 %). Figura:
`pr_curve_val.png`.

### Zona ambigua 0,4–0,6 → "incierta, requiere revisión humana"

En test se deriva a revisión el **0,56 %** de los casos (32). En esa franja el modelo acierta solo el
53 %, y la accuracy del resto sube de 0,9894 a **0,9919**. La curva completa de cobertura contra
accuracy está en `coverage_test.png`.

### Nivel de confianza

`confianza = max(p, 1 − p)` sobre la probabilidad calibrada.

- **baja**: la probabilidad cae en la zona ambigua (confianza < 0,6) o la entrada está fuera de
  distribución.
- **alta**: confianza ≥ **0,88**. Es el menor corte desde el cual la accuracy *local* (ventanas de
  0,10) en validación es ≥ 97 %.
- **media**: el resto.

| Nivel | Validación (fracción / accuracy) | Test (fracción / accuracy) |
|---|---|---|
| baja | 0,6 % / 0,595 | 0,6 % / 0,531 |
| media | 2,5 % / 0,794 | 2,5 % / 0,789 |
| alta | 96,9 % / 0,998 | 97,0 % / 0,997 |

> La probabilidad refleja **parecido con los textos de entrenamiento**, no certeza sobre la verdad de
> los hechos. En LIAR, el mismo modelo asigna confianza "alta" a afirmaciones que clasifica al azar (§8).

### Entradas fuera de distribución

La inferencia (`src/fakenews/predict.py`) advierte y responde "incierta" con confianza baja en estos
casos: texto vacío, solo enlaces, < 20 palabras, texto que no parece estar en inglés (proporción de
stopwords o de letras no ASCII) o con < 50 % de vocabulario conocido. Si el texto tiene > 3000
palabras, advierte y baja la confianza un nivel.

## 7. Experimento controlado de fuga (formulario 12)

Mismo modelo (LR con el mismo C y los mismos n-gramas), mismos datos, misma semilla y misma división.
Lo único que cambia es la limpieza.

Fuente: `reports/leakage_experiment_val.json` y `test_results.json` (`experimento_fuga_test`).

| | F1 falsa val | F1 falsa test | Accuracy test |
|---|---|---|---|
| **Antes** — limpieza mínima, con marcas | 0,9954 | 0,9934 | 0,9941 |
| **Después** — limpieza completa, sin marcas | 0,9851 | 0,9849 | 0,9864 |

**Interpretación.** Al quitar las marcas, el F1 cae unos 0,9 puntos. Con marcas, el modelo se apoyaba
en atajos: sus términos más influyentes eran `reuters` y `washington` (dateline) hacia "real", y
`via`, `video`, `image`, `featured` y `com` (créditos y enlaces) hacia "falsa". Sin marcas,
casi no pierde rendimiento porque **le quedan otras señales de estilo de la fuente**. Entre los
términos más influyentes del productivo están:

- Hacia "real": `said`, `on tuesday`, `on wednesday`, `president donald`, `on twitter`,
  `spokesman`, `reporters`. Son convenciones de redacción de agencia: atribución constante y día de
  la semana.
- Hacia "falsa": `gop`, `the us`, `breaking`, `just`, `watch`, `you`, `hillary`. Es registro de
  blog, informal y dirigido al lector.

Estos términos no son marcas de fuente en sentido estricto, pero **siguen siendo señales del estilo de
cada fuente**. Por eso la caída chica no prueba que el modelo use el contenido. La prueba más fuerte
es la validación externa.

Figura: `reports/figures/leakage_experiment.png`.

## 8. Validación externa en LIAR (sin reentrenar ni ajustar nada)

Fuente: `reports/external_liar.json`. 8090 afirmaciones (44 % falsas).

| Modelo | Accuracy [IC 95 %] | F1 falsa | F1 real | Recall falsa | p media |
|---|---|---|---|---|---|
| Trivial | 0,560 [0,548–0,571] | 0,000 | 0,718 | 0,000 | 0,453 |
| LR baseline | 0,511 [0,500–0,522] | 0,541 | 0,477 | 0,655 | 0,605 |
| **SVM (productivo)** | **0,487 [0,476–0,498]** | 0,570 | 0,364 | 0,772 | 0,710 |
| Bosque (estilo) | 0,536 | 0,372 | 0,632 | 0,312 | 0,383 |
| kNN coseno | 0,532 | 0,464 | 0,585 | 0,460 | 0,453 |
| MLP | 0,515 | 0,543 | 0,483 | 0,654 | 0,608 |

**Lectura (sin maquillar).** Ningún modelo supera al trivial en LIAR. El productivo empuja casi todo
hacia "falsa", porque las afirmaciones cortas y sin atribución ("said", día de la semana) no se
parecen al estilo Reuters. Tampoco mejora si se mira solo las afirmaciones de ≥ 20 palabras (2914
casos): accuracy 0,472.

Esto es una **evidencia de generalización** (negativa) y no un error de código. Parte de la caída se
explica por el cambio de dominio: LIAR tiene afirmaciones de 17 palabras de mediana, ISOT artículos
de 384. El resto, por el cambio de etiqueta: LIAR verifica afirmaciones, ISOT etiqueta por fuente.

El detector de entradas fuera de distribución marcaría el 64 % de LIAR como "texto muy corto", así
que la demo respondería "incierta" en esos casos. La zona ambigua por sí sola solo derivaría el 12 %.

Figura: `reports/figures/external_liar.png`.

## 9. Análisis de errores (objetivo 4)

Fuente: `reports/error_analysis.csv` y `error_analysis_summary.json`.

- 61 errores en test (34 FP y 27 FN). El CSV tiene **40** casos (20 FP y 20 FN, ordenados por
  confianza): 26 son de **alta confianza**.
- Cada caso trae texto resumido, probabilidad, términos que más empujaron, rasgos de estilo, una
  categoría tentativa y un **comentario marcado "BORRADOR – revisar por el equipo"**.
- Las columnas `comentario_equipo` y `revisado_por` están vacías a propósito: el acta exige que el
  **equipo** revise y comente al menos 20 casos.

Patrones visibles en el borrador, a confirmar por el equipo:

- **Etiqueta por fuente contra contenido.** Hay notas de Reuters que reproducen tuits de Trump
  ("Trump on Twitter (Aug 17)…", etiqueta real) y el modelo las predice falsas con p ≈ 1,00. El
  contenido tiene el tono de las falsas; lo que es "real" es la fuente.
- **Falsas con formato de agencia**: textos de sitios poco confiables que copian el estilo con muchas
  atribuciones y quedan del lado "real".
- **Opinión en primera persona**, textos basados en redes sociales y temas con vocabulario compartido
  entre clases.

### Desacuerdo entre texto y estilo (formulario 8, validación)

Fuente: `reports/disagreement_val.csv`.

- La LR de texto y el bosque de estilo discrepan en 429 casos (7,5 %). La LR acierta 381 y el bosque
  48; el ensamble acierta 390.
- Cuando acierta el modelo de estilo, los textos tienen más exclamaciones (0,32 contra 0,13 cada 100
  palabras) y más palabras en mayúsculas (4,1 % contra 1,4 %).
- El ensamble sube el F1 de falsa en validación de 0,9851 a 0,9869 y en test de 0,9849 a 0,9861,
  pero McNemar en test no es significativo (p = 0,31). La no linealidad aprovechable existe, pero es
  marginal. Es la limitación que anticipaba el acta: ambos modelos aprenden del mismo dataset y
  comparten el sesgo.

## 10. Diagnóstico del "ambiente informativo" (formulario 10, exploratorio)

Fuente: `reports/clustering.json`.

- k-means sobre LSA (TF-IDF reducido con SVD a 100 dimensiones). La silueta es muy baja en todo el
  rango (0,050 con k = 5 a 0,060 con k = 10) y el codo no muestra un quiebre claro: **los grupos se
  solapan mucho**. Se usa k = 10, la mejor silueta, que coincide con el borde de la grilla del acta.
- La proporción de falsas por grupo va de **2 %** (UE/Brexit) a **97 %** (un grupo de registro
  coloquial: "trump, you, that"). Los grupos de política de EE.UU. (Clinton, Rusia, policía) quedan
  en 38–55 %. Separar por tema no equivale a separar por veracidad, salvo donde el tema coincide con
  la fuente: los temas internacionales vienen casi todos de Reuters.
- Anomalías: 134 noticias por encima del percentil 99,5 de distancia al centro más cercano (78 %
  falsas).
- La GMM asigna una pertenencia ambigua (< 0,6) al 5,9 % de los textos.
- Eje temporal: la proporción de falsas por trimestre refleja cómo se armó el dataset (rangos de
  fechas distintos por clase), no la evolución real de la desinformación.

Figuras: `kmeans_elbow.png`, `kmeans_silhouette.png`, `kmeans_fake_share.png`.

## 11. Componentes adicionales de los laboratorios

- **Regresión logística desde cero** (labs 07 y 11–13; `reports/scratch_vs_sklearn.json`):
  - Con descenso de gradiente de 300 épocas, la tasa 0,5 llega a F1 0,850, la tasa 5 a 0,936 y la
    tasa 50 a 0,952.
  - Coincide con sklearn en el 97,9 % de las predicciones.
  - La figura `scratch_learning_rates.png` muestra cómo la tasa de aprendizaje cambia la velocidad
    de convergencia.
- **Curva de la red** (lab 13; `mlp_learning_curve.png`): la pérdida de train cae a 0,0006 y la de
  validación se estanca en 0,036. Es el patrón típico de memorización: F1 de train 1,000.
- **Escalas en kNN** (lab 09): sobre los rasgos de estilo, kNN sin estandarizar logra F1 0,836 y
  estandarizado 0,909.
- **Algoritmo genético** (labs 18–19; `genetic_feature_selection.json`): eligió 8 de 12 rasgos de
  estilo, con F1 val 0,896 contra 0,894 usando todos. Es exploratorio: el fitness usa validación, así
  que la cifra es optimista y no se usó para el modelo productivo.
- **Trazabilidad** (lab 17): la demo muestra las 3 noticias de entrenamiento más parecidas (coseno).
  La zona ambigua funciona como regla de abstención.

## 12. Limitaciones y riesgos éticos

### Limitaciones

- **El modelo aprende el estilo de la fuente.** Rinde 98,9 % en ISOT y está por debajo del azar en
  LIAR. No verifica hechos: detecta parecido con el estilo de los textos de entrenamiento.
- **Sesgo de período y tema**: política de EE.UU. entre 2015 y 2018. No sirve para noticias nuevas,
  de otros países o de otros temas.
- **Sátira y opinión** quedan dentro de "falsa"; hay noticias reales con tono sensacionalista (tuits
  citados) que se marcan como falsas con alta confianza.
- **Solo inglés**: no es extrapolable al español.
- El léxico sensacionalista y las reglas de limpieza son artesanales. Pueden tener sesgos propios.

### Riesgos

Puntaje = probabilidad × impacto, de 1 a 5 cada uno. Es la regla del lab 20: ALTO si el puntaje es
≥ 16, MEDIO si es ≥ 8.

| Riesgo | P | I | Nivel | Mitigación |
|---|---|---|---|---|
| Falso positivo que desacredita a un medio legítimo | 3 | 5 | **MEDIO (15)** | presentar siempre como estimación; zona ambigua; nunca usar como veredicto |
| Falso negativo que da falso respaldo a una falsa ("la herramienta dice real") | 4 | 5 | **ALTO (20)** | aviso permanente; recall de falsa priorizado; advertencia de entradas fuera de distribución |
| Uso como único criterio de moderación o censura | 3 | 5 | **MEDIO (15)** | documentado como uso prohibido; supervisión humana obligatoria |
| Generalización nula fuera de ISOT (demostrada en LIAR) | 5 | 4 | **ALTO (20)** | declarado en la demo y en el informe; reentrenar con datos diversos |
| Sesgo político (registro coloquial o de un sector asociado a "falsa") | 4 | 4 | **ALTO (16)** | auditar los términos influyentes; no usar para perfilar medios ni personas |
| Confianza alta mal interpretada como certeza | 4 | 4 | **ALTO (16)** | texto explícito: "parecido con el entrenamiento, no verdad" |

**Rama de visión (formulario 15): no incluida.** La única extensión razonable sería pasar por OCR las
capturas o memes y usar el clasificador actual. Se descartó por plazo y alcance; además, los errores
del OCR contaminarían la entrada.

## 13. Evaluación contra los criterios de éxito del acta

Fuente: `reports/success_criteria.json`.

| Criterio | Resultado | Estado |
|---|---|---|
| Accuracy en test > 80 % | 0,989 | Cumplido (dentro de ISOT) |
| F1 por clase > 0,80 | real 0,990 · falsa 0,988 | Cumplido |
| Supera con claridad al clasificador ingenuo, sin favorecer una clase | trivial 0,547 / F1 falsa 0 | Cumplido |
| Modelo adicional ≥ baseline, con la diferencia explicada | SVM 0,988 frente a LR 0,985; McNemar p = 0,030 (§4) | Cumplido |
| Train/val/test sin filtración | 0 solapamientos exactos y casi-duplicados | Cumplido |
| ≥ 20 errores revisados y comentados | 40 casos con borrador; 0 comentados por el equipo | **Pendiente (equipo)** |
| Demo devuelve etiqueta y probabilidad | verificado por los tests y la CLI | Cumplido |
| Accuracy > 98 % ⇒ sospecha | **alerta activa**: diagnósticos sin tokens de fuente ni marcas residuales, pero LIAR muestra que no generaliza | Se declara |

## 14. Conclusiones

1. Con datos limpios y deduplicados, un modelo lineal sobre TF-IDF separa casi perfecto las dos
   fuentes de ISOT. El SVM mejora apenas al baseline y la red no aporta nada.
2. Sacar las marcas evidentes de fuente baja el rendimiento poco, pero el modelo sigue usando el
   estilo de redacción de cada fuente. **La validación externa muestra que no detecta veracidad.**
3. Lo más valioso del proyecto es la **metodología**: auditoría de fuga medida, test sagrado,
   calibración, abstención y validación externa. Muestra por qué un 99 % no alcanza para confiar en
   un detector.
4. Próximos pasos: entrenar con datasets cuya etiqueta sea de veracidad (LIAR, FakeNewsNet), mezclar
   varias fuentes en cada clase y evaluar con validación cruzada por fuente.

---

## 15. Iteración 2: generalización real

**Objetivo**: mejorar la generalización y la honestidad de la confianza, **no** el accuracy dentro de
ISOT. Detalle del proceso y de las decisiones en `docs/01_iteracion_2.md`. Las cifras salen de
`reports/*_v2.*`, `reports/panel_v2.csv`, `reports/final_v2.json` y
`reports/comparison_it1_vs_it2.csv`.

### 15.1 Holdout y reglas de evaluación

- **LIAR ya no tiene ninguna porción virgen**: en la iteración 1 se evaluaron las tres juntas, aunque
  el log no lo registró.
- Por eso se agregó **WELFake** (CC BY 4.0) como holdout externo de nivel artículo. Hubo que
  verificar dos cosas: la dirección de las etiquetas (la descripción de Zenodo está invertida respecto
  del CSV) y que el 61 % de WELFake era ISOT, que se eliminó. Quedan 23 250 artículos nuevos:
  11 625 de desarrollo y 11 625 de holdout.
- Todas las decisiones se tomaron con ISOT val, las vistas de cambio de grupo y tiempo (pool
  train+val), WELFake-dev y LIAR-valid.
- El ISOT test y el WELFake-holdout se tocaron **una vez** al final. El log registra 5 accesos de la
  iteración 2, sin re-accesos.

### 15.2 Auditoría de atajos de un solo rasgo

Fuente: `reports/shortcut_audit_v2.csv`, figura `shortcut_audit_v2.png`. Valores en AUC orientada,
medida en ISOT val.

| Rasgo (solo) | Crudo | v1 | v2 | v3 | Transfiere a WELFake-dev* |
|---|---|---|---|---|---|
| `subject` / formato de fecha (no entran al modelo) | 1,000 | — | — | — | — |
| dateline `(Reuters)` | 0,996 | 0,500 | 0,500 | 0,500 | 0,501 |
| verbos de atribución | 0,869 | 0,866 | 0,869 | **0,556** | **0,747** |
| `said/says/told` | 0,863 | 0,863 | 0,866 | **0,500** | **0,767** |
| palabras en mayúsculas en el título | 0,863 | 0,794 | 0,729 | 0,582 | 0,586 |
| días de la semana | 0,857 | 0,856 | **0,500** | 0,500 | 0,714 |
| apóstrofo como espacio (`don t`) | 0,847 | 0,795 | **0,500** | 0,500 | 0,500 |
| comillas curvas | 0,762 | **0,500** | 0,500 | 0,500 | 0,654 |
| `U.S.` con puntos | 0,755 | 0,752 | **0,500** | 0,500 | 0,454 |
| segunda persona (`you`) | 0,733 | 0,732 | 0,732 | 0,731 | 0,527 |
| signos de pregunta | 0,726 | 0,719 | 0,719 | 0,719 | 0,556 |
| tokens neutros de enlaces (`xurlx`…) | — | 0,500 | **0,630** | 0,500 | 0,500 |

\* AUC en WELFake-dev usando la orientación aprendida en ISOT. Un valor > 0,5 significa que el rasgo
también predice fuera de ISOT.

**Lectura.**

- La v1 ya neutralizaba las marcas grandes, pero dejaba la atribución, los días, el apóstrofo y `U.S.`.
- La v2 elimina estos últimos, aunque **introduce** un atajo nuevo: los tokens de enlace.
- La v3 enmascara la atribución, pero **`said` también predice en WELFake** (0,77): no es solo un
  atajo de ISOT, también es una señal de estilo periodístico que generaliza en parte. Enmascararlo
  resultó contraproducente (§15.3).

### 15.3 Panel de evaluación y escalera de ablación

Fuente: `reports/panel_v2.csv`, `ablation_v2.csv`, figura `ablation_v2.png`.

**Vistas:**

- **In-domain**: entrena en ISOT train y evalúa en ISOT val.
- **Grupo**: 4 folds; cada uno retiene un `subject` real y un grupo de `subject` falsos.
- **Tiempo**: entrena antes del 2017-07-01 y evalúa después.
- **Externo-dev**: WELFake-dev.
- **LIAR-valid**: se reporta, pero no entra al puntaje porque D1 entrena con LIAR.
- **Invarianza**: 5 transformaciones que no cambian la veracidad, sobre 2000 textos de val.

**Puntaje robusto** = promedio de F1 macro en grupo, tiempo y WELFake-dev. Pisos: F1 por clase
in-domain ≥ 0,90 e invarianza ≤ 2 %. Todos los candidatos pasaron los pisos.

| Candidato | In (F1 falsa) | Grupo | Tiempo | WELFake-dev | LIAR-valid | Invar. máx. | **Robusto** |
|---|---|---|---|---|---|---|---|
| **it1: SVM palabras v1** (referencia) | 0,989 | 0,739 | 0,905 | 0,520 | 0,504 | 0,55 % | 0,721 |
| LR palabras v1 C = 10 | 0,980 | 0,730 | 0,897 | 0,546 | 0,535 | 0,65 % | 0,724 |
| LR palabras v1 C = 1 | 0,972 | 0,720 | 0,851 | 0,549 | 0,561 | 0,50 % | 0,707 |
| LR palabras v2 C = 10 | 0,981 | 0,711 | 0,832 | 0,525 | 0,554 | 0,05 % | 0,689 |
| LR palabras v3 C = 10 | 0,982 | 0,677 | 0,787 | 0,512 | 0,586 | 0,00 % | 0,659 |
| SVM v2 / v3 | 0,989 / 0,987 | 0,703 / 0,676 | 0,861 / 0,809 | 0,494 / 0,484 | 0,529 / 0,553 | ≤ 0,1 % | 0,686 / 0,656 |
| LR caracteres v1 | 0,990 | 0,744 | 0,849 | 0,594 | 0,550 | 0,30 % | 0,729 |
| **LR caracteres v2 (elegido)** | **0,988** | 0,737 | **0,896** | **0,594** | 0,567 | 0,15 % | **0,742** |
| LR caracteres v3 | 0,987 | 0,709 | 0,842 | 0,580 | 0,567 | 0,15 % | 0,710 |
| Ensamble (texto + estilo) v2 / v3 | 0,983 / 0,981 | **0,749** / 0,690 | 0,862 / 0,805 | 0,528 / 0,509 | 0,553 / 0,571 | ≤ 0,5 % | 0,713 / 0,668 |
| D1: LR v2 + LIAR-train (peso 0,25) | 0,981 | 0,749 | 0,898 | 0,533 | *0,669* | 0 % | 0,726 |
| D1: LR v2 + LIAR-train (peso 1) | 0,974 | 0,746 | 0,883 | 0,530 | *0,652* | 0 % | 0,720 |
| D1: LR v3 + LIAR-train (pesos 0,25 / 1) | 0,979 / 0,971 | 0,715 / 0,711 | 0,862 / 0,844 | 0,518 / 0,522 | *0,667 / 0,665* | 0 % | 0,698 / 0,693 |

Trivial en WELFake-dev: accuracy 0,583, F1 macro 0,368. En LIAR-valid: 0,526. *En cursiva*: LIAR-valid
para D1, que no es externo para esos modelos.

**Qué se aprendió (incluidas las decisiones que salieron peor):**

1. **Neutralizar el texto no mejora a los modelos de palabras.** v1 → v2 → v3 baja el puntaje robusto
   en todas las familias de palabras (LR 0,724 → 0,689 → 0,659; SVM 0,721 → 0,686 → 0,656). Sin los
   atajos evidentes, el modelo encuentra otros, y además se pierde señal que sí generalizaba (`said`).
2. **Los n-gramas de caracteres son lo único que mejora afuera**: WELFake-dev sube de 0,52–0,55 a
   0,59. Con caracteres, v2 ayuda en la vista temporal (0,849 → 0,896), porque quita días y fechas,
   que son un atajo temporal. v3 vuelve a empeorar.
3. **Regularización más fuerte** (C = 1 frente a C = 10): +0,003 en WELFake y −0,05 en la vista
   temporal. No es una mejora robusta.
4. **El ensamble con rasgos de estilo** es el mejor en la vista de grupo (0,749), pero el peor afuera.
   Los rasgos de estilo son atajos de fuente.
5. **D1 (sumar LIAR-train)** mejora LIAR-valid (0,55 → 0,67), que para D1 ya no es externo, y **no
   mejora WELFake** (0,53). Aprender las afirmaciones cortas de PolitiFact no transfiere a artículos.
   No se adopta.
6. kNN y MLP no se reentrenaron porque en la iteración 1 no aportaron.

**Modelo final elegido por el criterio**: LR sobre n-gramas de caracteres (`char_wb`, 3–5 gramas,
150 000 rasgos, C = 10), con limpieza v2 y calibración isotónica (elegida por Brier en val: 0,0081).

Términos más influyentes del modelo final (`final_v2.json → top_terminos_v2`):

- Hacia "falsa": `.th`, `.the`, `s.t`, `this`, `gop`, `just`. Los primeros son **oraciones pegadas sin
  espacio** después del punto, un artefacto del scraping de los blogs que la auditoría no había
  cubierto.
- Hacia "real": `said`, ` us `, ` . `, ` , `, `...`. Los signos con espacios alrededor son artefactos
  de la propia limpieza (huecos que quedan al quitar palabras).

Medido en val, normalizar el espaciado de oraciones cambia el 1,25 % de las predicciones (0 % en el
modelo de la iteración 1). En una submuestra de 400 textos dio 2,5 %. **Está en el límite del criterio
de invarianza**: es un atajo de formato residual, declarado y con su test
(`test_sentence_spacing_dependence_is_bounded`).

### 15.4 Confianza y abstención (OOD)

Fuente: `reports/train_v2.json`, `shrink_alternatives_v2.json`; figuras `abstention_coverage_v2.png`
y `reliability_shift_v2.png`.

**Puntaje OOD.** Es una logística sobre 4 rasgos baratos:

- similitud coseno máxima con las noticias de entrenamiento,
- media de los 5 vecinos,
- cobertura de vocabulario,
- largo.

Se ajusta para separar ISOT-val del desarrollo externo (WELFake-dev + LIAR-valid). La probabilidad se
**contrae hacia la prior** (0,453) en proporción al puntaje. Si el puntaje es ≥ τ, el sistema responde
"incierta". Si es ≥ τ_cap (0,756), la confianza no puede ser "alta".

**Elección de τ, solo con desarrollo.** Ningún τ logra que, sobre lo respondido en el externo de
desarrollo, el modelo sea **no peor que el trivial** manteniendo al menos el 95 % de cobertura en
ISOT-val. Sobre lo respondido, el modelo acierta alrededor de 0,59 para casi cualquier τ: **el
puntaje OOD no se correlaciona con los aciertos fuera de ISOT**. Solo con τ ≤ 0,25 lo respondido
supera al trivial, pero entonces se responde menos del 39 % de ISOT. Por regla se tomó **τ = 0,80**:
responde el 97,5 % de ISOT-val y se abstiene en el 27 % de WELFake-dev y el 6 % de LIAR-valid.

**Calibración en las vistas con cambio de distribución** (sin contraer, modelo final). ECE de 0,237 en
grupo y 0,117 en tiempo, contra 0,003 in-domain. La confianza in-domain no se traslada a otros
grupos ni a otros períodos.

**Intercambio de la contracción** (medido solo en val y desarrollo):

| ECE | Sin contracción | Proporcional (adoptada y evaluada) | "Solo exceso" (alternativa, no evaluada en holdout) |
|---|---|---|---|
| ISOT val | 0,003 | **0,181** | 0,007 |
| WELFake-dev | 0,344 | **0,102** | 0,282 |
| LIAR-valid | 0,272 | **0,109** | 0,265 |

La contracción proporcional hace la probabilidad mucho más honesta afuera, pero **subconfiada
adentro**. Este efecto in-domain **ya era visible en validación y se pasó por alto antes de la
evaluación final**; se declara como error del proceso. Se mantiene la proporcional por dos razones:
es la que se evaluó una sola vez y es la opción conservadora (subestimar la confianza es menos
dañino que sobreestimarla). Usar la alternativa requeriría una nueva evaluación en holdout.

### 15.5 Evaluación final única

Fuentes: `reports/final_v2.json` y `comparison_it1_vs_it2.csv`; figuras `confusion_v2_*.png` y
`reliability_final_v2.png`.

| Conjunto | Métrica | Iteración 1 | **Iteración 2** |
|---|---|---|---|
| ISOT test | accuracy (forzado) | 0,9894 | **0,9899** [IC 95 % 0,987–0,992] |
| ISOT test | F1 falsa / F1 real | 0,988 / 0,990 | 0,989 / 0,991 |
| **WELFake holdout** | accuracy (forzado); trivial 0,583 | 0,547 | **0,581** [0,571–0,589] |
| WELFake holdout | F1 macro; trivial 0,368 | 0,525 | **0,575** |
| WELFake holdout | ECE (it2: contraída) | 0,398 | **0,101** |
| WELFake holdout | Brier (it2: contraída); trivial 0,244 | 0,397 | **0,242** |
| WELFake holdout | cobertura del sistema (abstención OOD + zona ambigua) | 0,951 | 0,513 |
| WELFake holdout | accuracy de lo respondido / trivial sobre lo respondido | 0,549 / 0,574 | 0,630 / 0,631 |
| LIAR test (**no virgen**) | accuracy; trivial 0,574 | 0,473 | 0,514 |

**Lectura honesta:**

- **In-domain no bajó**: 0,989 → 0,990. Sigue arriba del 98 %, así que **la alerta del acta se
  mantiene**. El modelo sigue apoyándose en el estilo de redacción de las fuentes de ISOT (`said`,
  espaciado de oraciones).
- **La generalización mejoró, pero no alcanza.** En WELFake-holdout la accuracy forzada sube 3,4
  puntos y el F1 macro 5 puntos. Aun así **queda por debajo del trivial** (0,581 contra 0,583).
- Con abstención, el sistema responde el 51 % de WELFake con accuracy 0,630. Eso **empata, no
  supera**, al trivial sobre esos mismos casos (0,631). Es decir: la abstención **lleva al sistema al
  nivel del trivial, no por encima**.
- Afuera, el error dominante es marcar como **falsas** noticias reales de medios no-Reuters: 4078
  falsos positivos contra 796 falsos negativos en el holdout.
- **La honestidad de la probabilidad sí mejoró mucho** (ECE 0,40 → 0,10; Brier al nivel del trivial).
  El sistema ya no se muestra "muy seguro" mientras falla afuera. El costo es la subconfianza
  in-domain (§15.4).
- En ISOT test, el sistema responde el 92,7 % (se abstiene por OOD en el 2,8 % y por zona ambigua en
  el 4,5 %) con accuracy 0,995 sobre lo respondido. Por nivel de confianza: "alta" acierta 0,999,
  "media" 0,993 y "baja" 0,898.

### 15.6 Análisis de errores v2

Fuente: `reports/error_analysis_v2.csv`, el archivo que **debe revisar el equipo**.

- 40 errores del modelo final: 20 de ISOT test y 20 de WELFake-holdout, con FP y FN balanceados.
- Todos son de alta confianza cruda. En 9 de ellos el sistema se habría abstenido por OOD.
- Cada caso tiene una categoría tentativa y un comentario "BORRADOR – revisar por el equipo".
- El archivo de la iteración 1 (`error_analysis.csv`) no se modificó.

### 15.7 Criterios de éxito (reevaluados con el modelo v2)

Fuente: `reports/success_criteria_v2.json`.

| Criterio | Estado |
|---|---|
| Accuracy > 80 % (ISOT test) | Cumplido (0,990), **solo dentro de ISOT** |
| F1 por clase > 0,80 | Cumplido (0,991 / 0,989) |
| Supera claramente al trivial | Cumplido en ISOT; **no** en WELFake ni en LIAR |
| Modelo adicional ≥ baseline, con la diferencia explicada | Cumplido (iteración 1: McNemar p = 0,030; el v2 se elige por robustez, §15.3) |
| Sin filtración | Cumplido (ISOT sin solapamientos; WELFake deduplicado contra todo ISOT y LIAR) |
| ≥ 20 errores revisados | **Pendiente (equipo)**: `error_analysis_v2.csv` |
| Demo con etiqueta y probabilidad | Cumplido |
| Accuracy > 98 % ⇒ sospecha | **Alerta activa** y justificada por los resultados externos |

### 15.8 Limitaciones actualizadas

- El modelo aprende **patrones de redacción de las fuentes del dataset**: atribución, registro y
  formato, incluido el espaciado de oraciones. **No verifica hechos.**
- La probabilidad es **parecido con el entrenamiento**, no certeza. Con la contracción OOD es más
  honesta afuera, pero subconfiada en textos tipo ISOT.
- Fuera de ISOT el sistema **no supera al trivial**, ni siquiera respondiendo solo los casos que
  considera dentro de distribución.
- Ninguna de las vías probadas resuelve la causa raíz: en ISOT, clase = fuente. Eso incluye
  neutralizar el texto, regularizar más, el ensamble y sumar LIAR. El arreglo de fondo exige datos
  donde **ambas clases vengan de muchas fuentes de nivel artículo**, y evaluar con **validación
  cruzada por fuente**.
- Nunca usar como único criterio de moderación.
