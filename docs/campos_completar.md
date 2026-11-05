# Valores reales para los campos `[completar]` del acta

Cada valor sale de una corrida real. Entre paréntesis va el archivo de `reports/` de donde se tomó
(semilla 42, 04/11/2026). Los textos están listos para pegar en los formularios.

## Equipo y roles (sección "Equipo y roles")

| Rol | Responsable |
|---|---|
| Coordinación | pendiente: equipo |
| Datos | pendiente: equipo |
| Modelado | pendiente: equipo |
| Evaluación | pendiente: equipo |
| Demo y documentación | pendiente: equipo |

## Formulario 3 · Auditar los datos

| Campo | Valor |
|---|---|
| Faltantes detectados | Sin nulos en ninguna columna. 630 falsas tienen el cuerpo vacío (solo título) y 1 real; 1105 falsas y 1 real tienen < 20 palabras; 84 falsas son solo enlaces; 10 fechas de falsas no se pueden leer (son URLs en la columna `date`). (`data_audit.json`) |
| Valores atípicos | Cuerpos de hasta 8375 palabras (28 falsas y 1 real con > 5000). Duplicados exactos: 6026 cuerpos falsos y 225 reales. Tras limpiar se eliminaron 729 textos cortos, 5871 duplicados exactos y 99 casi-duplicados (MinHash, Jaccard ≥ 0,8). (`data_audit.json`, `splits_manifest.json`) |
| Desbalance o representatividad | Crudo: 23 481 falsas y 21 417 reales. Tras deduplicar: 17 315 falsas (45,3 %) y 20 884 reales (54,7 %). Todo es política de EE.UU. y noticias internacionales de 2015 a 2018, y las reales son todas de Reuters. (`splits_manifest.json`) |

## Formulario 4 · Primer baseline

| Campo | Valor |
|---|---|
| Resultado inicial | Trivial (clase mayoritaria): accuracy 0,547, F1 falsa 0,000. Baseline TF-IDF + regresión logística (C = 100, unigramas), en validación: accuracy 0,987, F1 falsa 0,985 (precisión 0,987, recall 0,983), F1 real 0,988. En test: accuracy 0,986, F1 falsa 0,985 [IC 95 % 0,981–0,988], F1 real 0,988. (`val_results.json`, `test_results.json`) |

## Formulario 5 · Variable continua

| Campo | Valor |
|---|---|
| Error | Baseline "proporción de falsas de train" (p = 0,453): Brier 0,248, log-loss 0,689. LR TF-IDF: Brier 0,0104, log-loss 0,039. Modelo productivo (SVM calibrado): Brier 0,0082, log-loss 0,032, ECE 0,0041 (todo en test). (`test_results.json`) |

## Formulario 8 · No linealidad

| Campo | Valor |
|---|---|
| Valor del ensamble | El stacking (LR de texto + bosque de estilo) sube el F1 de falsa de 0,9851 a 0,9869 en validación y de 0,9849 a 0,9861 en test, pero no es significativo (McNemar p = 0,31). En los 429 desacuerdos de validación acierta la LR en 381, el bosque en 48 y el ensamble en 390. Hay una no linealidad aprovechable, pero es marginal. (`val_results.json`, `test_results.json`) |

## Formulario 12 · Experimento controlado

| Campo | Valor |
|---|---|
| Resultado antes | Misma LR con limpieza mínima (con marcas de fuente y formato): F1 falsa 0,9934 y accuracy 0,9941 en test (0,9954 en validación). (`test_results.json`, `leakage_experiment_val.json`) |
| Resultado después | Limpieza completa (sin marcas): F1 falsa 0,9849 y accuracy 0,9864 en test (0,9851 en validación). Cae unos 0,9 puntos. El modelo sigue apoyándose en el estilo de redacción de cada fuente (`said`, días de la semana frente a `gop`, `just`, `watch`). En LIAR, el modelo productivo baja a accuracy 0,487. (`external_liar.json`) |

## Formulario 13 · Cerrar datos, baseline y experimento

| Campo | Valor |
|---|---|
| Configuración del experimento | TF-IDF (sublinear_tf, min_df = 3, max_df = 0,9, hasta 100 000 términos) + regresión logística (lbfgs). Grilla C ∈ {0,1; 0,3; 1; 3; 10; 30; 100} × n-gramas ∈ {(1,1), (1,2)}. Datos limpios y deduplicados; 70/15/15 estratificado con semilla 42 (26 739 / 5730 / 5730). (`baseline_logreg_grid.csv`) |
| Métrica observada | Validación: F1 falsa 0,985, accuracy 0,987, precisión 0,987, recall 0,983. Test: F1 falsa 0,985, accuracy 0,986, precisión 0,988, recall 0,982. |
| Comportamiento de entrenamiento | F1 falsa en train según C (unigramas): 0,954 (0,1), 0,970 (0,3), 0,984 (1), 0,991 (3), 0,999 (10), 0,999 (30), 0,9995 (100). Con poca regularización el modelo casi memoriza train. |
| Comportamiento de validación | F1 falsa en validación (unigramas): 0,949, 0,962, 0,972, 0,978, 0,983, 0,985, 0,985. Con bigramas: 0,954, 0,965, 0,972, 0,978, 0,980, 0,981, 0,981. |
| Configuración elegida | C = 100, unigramas: F1 val 0,9851 con brecha train–val de 0,014 (tope tolerado 0,03). Desde C = 30 hay una meseta (0,9846), así que el resultado no depende de estar en el borde de la grilla. |
| Síntesis Entrega 2 (resultados reales) | Baseline LR con F1 falsa de 0,985 en test frente a 0 del trivial. El experimento de fuga muestra una caída de 0,9 puntos al quitar marcas. El SVM lineal mejora al baseline (F1 0,988; McNemar p = 0,030). La validación en LIAR (accuracy 0,487) indica que el modelo aprende el estilo de las fuentes de ISOT y no generaliza a otro dominio. |

## Formulario 14 · ¿Una red agrega valor?

| Campo | Valor |
|---|---|
| No linealidad | El ensamble y el MLP no mejoran de forma relevante al lineal. MLP (32 neuronas ReLU) frente a la LR: +0,0008 de F1 en validación y +0,00003 en test (McNemar p = 1,0). El árbol y el bosque de estilo solos rinden bastante menos (F1 0,890 y 0,919 en test). Hay poca no linealidad que aprovechar. **Decisión: no adoptar la red.** (`test_results.json → decision_red`) |

## Otros campos que dependen del equipo

- Fecha real de entrega (el acta supone el 18/11/2026): pendiente: equipo.
- Firmas de aprobación: pendiente: equipo.
- Revisión humana de ≥ 20 errores (`reports/error_analysis_v3.csv`, columnas `comentario_equipo` y
  `revisado_por`): pendiente: equipo.

---

## Iteración 2 (para actualizar el acta si los docentes lo piden)

Fuentes: `reports/panel_v2.csv`, `reports/final_v2.json`, `reports/comparison_it1_vs_it2.csv`.

| Formulario · campo | Valor nuevo |
|---|---|
| 3 · Riesgo de fuga | Además de las marcas de la iteración 1, la auditoría de un solo rasgo encontró: verbos de atribución (AUC 0,87), días de la semana (0,86), apóstrofo como espacio (0,85), comillas curvas (0,76), `U.S.` con puntos (0,75) y, en el modelo final, el espaciado de oraciones. (`shortcut_audit_v2.csv`) |
| 12 · Experimento controlado (ablación v1 → v2 → v3) | LR de palabras, puntaje robusto: 0,724 → 0,689 → 0,659. LR de caracteres: 0,729 → **0,742** → 0,710. Neutralizar más no siempre generaliza mejor. (`ablation_v2.csv`) |
| 12 · Próximo paso (resultado) | Validación externa nueva (WELFake-holdout, deduplicado contra ISOT): accuracy 0,581 (it2) frente a 0,547 (it1); trivial 0,583 |
| 13 · Configuración elegida (it2) | LR sobre n-gramas de caracteres `char_wb` 3–5 (150 000 rasgos), C = 10, limpieza v2, calibración isotónica, umbral 0,50, zona 0,4–0,6, OOD τ = 0,80 |
| 14 · No linealidad (it2) | Ensamble con estilo: el mejor en la vista de grupo (0,749), el peor afuera (WELFake-dev 0,528). Se mantiene un modelo lineal. |
| 8 · Valor del ensamble (it2) | No supera al lineal de caracteres en el puntaje robusto (0,713 frente a 0,742) |

---

## Cierre (iteración 3): valores vigentes

Si los docentes piden el estado final, estos valores reemplazan a los anteriores. Las tablas se generan
desde `reports/`.

**Corrección importante.** Lo escrito arriba para la iteración 2 sobre "validación externa" y
"mejora" se reinterpretó: la mejora en WELFake fue sobre todo un corrimiento de sesgo (informe §4).

| Formulario · campo | Valor vigente |
|---|---|
| 3 · Riesgo de fuga | Además de lo anterior: oración pegada sin espacio tras el punto (AUC 0,94 como rasgo único, aun después de `v2`). (`format_shortcut_discovery_v3.csv`) |
| 12 · Próximo paso (resultado final) | Ver la tabla de abajo: en FA-KES, holdout virgen con etiqueta de veracidad, la AUC es de azar |
| 13 · Configuración elegida (final) | LR de n-gramas de caracteres `char_wb` 3–5, C = 10, limpieza `v2`, isotónica, umbral 0,50, zona 0,4–0,6, compuerta de dominio combinada (retención 97 %) |
| 14 · Decisión | Se mantiene un modelo lineal. Embeddings congelados tampoco mejoran (puntaje robusto 0,623 frente a 0,742). |
| 15 · Visión | No incluida |

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

<!-- BEGIN GENERATED:criterios -->
| Criterio del acta | Estado |
|---|---|
| Accuracy en test > 80 % | cumplido (solo dentro de ISOT) |
| F1 por clase > 0,80 | cumplido (solo dentro de ISOT) |
| Supera con claridad al ingenuo sin favorecer una clase | cumplido en ISOT; ver externos |
| Modelo adicional iguala o supera al baseline, con la diferencia explicada | cumplido |
| Sin filtración entre splits | cumplido |
| >= 20 errores revisados por el equipo | pendiente: equipo (error_analysis_v3.csv) |
| Demo devuelve etiqueta y probabilidad | cumplido (si pytest pasa) |
| Accuracy > 98 % => sospecha | alerta cerrada con evidencia: ver informe |
<!-- END GENERATED:criterios -->
