# 01 · Iteración 2: generalización real

Objetivo: mejorar la **generalización** y la **honestidad de la confianza**, no el accuracy dentro de
ISOT. Todas las cifras salen de `reports/` (semilla 42).

## (a) Estado verificado al empezar

| Ítem | Verificado en | Resultado |
|---|---|---|
| Commits de la iteración 1 | `git log` | 7 commits del equipo sobre `dev`; árbol limpio |
| ISOT test, productivo (SVM calibrado) | `reports/test_results.json` | accuracy 0,9894 · F1 falsa 0,9883 · F1 real 0,9903 (coincide con lo reportado) |
| LIAR | `reports/external_liar.json` | accuracy 0,4869 contra 0,5598 del trivial (coincide) |
| Log de accesos al test | `reports/test_access_log.jsonl` | 11 accesos a ISOT test (uno por modelo), sin re-accesos |
| **Accesos a LIAR** | código de `evaluate_external` | **No quedaron registrados en el log**, pero el código concatenaba `train + valid + test`. **Las tres porciones de LIAR ya se miraron**: no queda ninguna virgen. |

**Consecuencia (sección 2.4 del prompt).** LIAR no puede servir como holdout final. Se usó un
**holdout externo nuevo de nivel artículo: WELFake** (Zenodo 4561253, CC BY 4.0):

- Se deduplicó (exacto y casi-duplicado) contra **todas** las filas de ISOT y contra LIAR, y se
  dividió 50/50 en `dev` (decisiones) y `holdout` (se toca una sola vez, al final).
- LIAR-valid se usa como desarrollo externo.
- LIAR-test se reporta, **marcado como "no virgen"**.

A partir de esta iteración, `evaluate_v2` también registra en el log los accesos a los conjuntos
externos.

### Hallazgos de la verificación de WELFake

- **La descripción de Zenodo dice "0 = fake, 1 = real", pero el CSV usa la convención opuesta.** Las
  44 284 filas que coinciden textualmente con ISOT tienen etiqueta 1 cuando en ISOT son falsas
  (acuerdo 99,97 %). En el CSV, `1 = falsa`, igual que nuestra convención, así que **no se invirtió
  nada**. `verify_welfake_label_direction` falla si esto cambia.
- **El 61 % de WELFake es ISOT** (44 284 de 72 134 filas). Tras quitar textos cortos, todo lo que
  coincide con ISOT o con LIAR y los duplicados internos, quedan **23 250 artículos**: 11 625 en dev
  y 11 625 en holdout, con 41,7 % de falsas. Solo el 0,1 % conserva un dateline de Reuters.

## (b) Plan y prioridades

1. **A. Auditoría de atajos** de un solo rasgo, sobre el texto crudo y sobre cada perfil (`shortcuts.py`).
2. **B. Perfiles de limpieza** v2 y v3, configurables y sin romper v1 (`cleaning.py`, `cleaning.profile`).
3. **Panel de evaluación** (`panel.py`): in-domain, cambio de grupo, cambio temporal, externo-dev
   (WELFake-dev), LIAR-valid e invarianza. La selección robusta se aplica igual a todos los candidatos.
4. **C/D. Candidatos**:
   - LR de palabras (C = 10 y C = 1) en v1, v2 y v3.
   - SVM calibrado en v1 (la configuración de la iteración 1), v2 y v3.
   - LR de n-gramas de caracteres en v2 y v3.
   - Ensamble con rasgos de estilo en v2 y v3.
   - **D1**: ISOT + LIAR-train con peso de dominio 0,25 o 1, en v2 y v3.
5. **E. OOD y abstención** (`ood.py`, `train_v2.py`): contraer la probabilidad hacia la prior y
   abstenerse, calibrado solo en desarrollo.
6. **F. Tests nuevos**; los 99 existentes siguen verdes.
7. **G. Evaluación final única** (`evaluate_v2.py`) en ISOT test y WELFake-holdout, registrada.
8. Documentación, README y comandos de git.

## (c) Decisiones tomadas sin consultar

| Decisión | Motivo |
|---|---|
| Holdout externo final = WELFake-holdout (50 % del WELFake deduplicado) | LIAR ya se miró completo en la iteración 1 |
| La vista 4 del puntaje robusto es **WELFake-dev**, no LIAR-valid | D1 entrena con LIAR-train: para D1, LIAR-valid deja de ser externo y el criterio no sería igual para todos. LIAR-valid se reporta igual. |
| WELFake se usa **solo para evaluar** (no se probó como dato de entrenamiento, D2) | Es el único externo de nivel artículo disponible; entrenar con él dejaría sin holdout independiente |
| Folds por grupo: 4 folds (`politicsNews`+`News`, `worldnews`+`politics`, y cada subject real con el grupo de subjects falsos chicos: `left-news`/`Government News`/`US_News`) | Las reales tienen 2 subjects y las falsas 5 muy desparejos; así cada subject queda afuera al menos una vez y cada fold tiene ambas clases |
| Corte temporal 2017-07-01 (pool train+val) | Deja ambas clases en los dos lados: 19 517 noticias para entrenar y 12 952 para evaluar |
| Invarianza: máximo sobre 5 transformaciones, en una muestra estratificada de 2000 textos de val | El criterio "≤ 2 %" se aplica a la peor transformación |
| v2 elimina días, meses, husos horarios y números (a token) y normaliza `U.S.` → `US` | La auditoría los marcó como atajos de redacción de agencia |
| v3 = v2 + enmascarar `said/told/officials/...`, quitar tokens de enlace, minúsculas en palabras TODO MAYÚSCULAS de 5+ letras | Justificado por la auditoría. Se prueba en el panel y no se asume que mejora: `said` también predice en WELFake. |
| El kNN y el MLP no se reentrenaron | En la iteración 1 no aportaron (MLP = LR; kNN muy por debajo) |
| D1 solo con LR | Es el único modelo del panel que acepta pesos por muestra dentro del pipeline sin cambios |
| El artefacto v2 se guarda aparte (`models/final_model_v2.joblib`) y la iteración 1 queda intacta | Preservar la referencia; `predict` usa el v2 si existe (`config.yaml → inference.artifact`) |

## (d) Resultados (resumen; detalle en `docs/informe_resultados.md` §15)

| | Iteración 1 | Iteración 2 |
|---|---|---|
| Modelo | SVM palabras 1–2 gramas, limpieza v1 | **LR n-gramas de caracteres 3–5, limpieza v2**, isotónica + OOD |
| Puntaje robusto (panel, desarrollo) | 0,721 | **0,742** |
| ISOT test accuracy | 0,989 | 0,990 |
| WELFake-holdout accuracy / F1 macro (trivial 0,583 / 0,368) | 0,547 / 0,525 | **0,581 / 0,575** |
| WELFake-holdout ECE | 0,398 | **0,101** |
| LIAR test (no virgen) | 0,473 | 0,514 |

**Hallazgos tardíos, declarados:**

1. **Atajo de espaciado de oraciones** (`.the`): apareció en los términos del modelo final. Cambia el
   1,25 % de las predicciones en val: está en el límite del criterio.
2. **La contracción OOD descalibra in-domain** (ECE en val 0,003 → 0,181). Era visible en validación
   antes de la evaluación final y lo pasé por alto. Se conserva la versión evaluada; la alternativa
   está en `reports/shrink_alternatives_v2.json`.

**Archivo de errores que debe revisar el equipo**: `reports/error_analysis_v2.csv`.
