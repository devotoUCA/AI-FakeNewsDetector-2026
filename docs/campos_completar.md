# Valores reales para los campos `[completar]` del acta

Cada valor sale de una corrida real. Entre paréntesis va el archivo de `reports/` de donde se tomó
(semilla 42, 04/11/2026). Los textos están listos para pegar en los formularios.

## Equipo y roles (sección "Equipo y roles")

| Rol | Responsable |
|---|---|
| Coordinación | **pendiente: lo define el equipo** |
| Datos | **pendiente: lo define el equipo** |
| Modelado | **pendiente: lo define el equipo** |
| Evaluación | **pendiente: lo define el equipo** |
| Demo y documentación | **pendiente: lo define el equipo** |

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

- Fecha real de entrega (el acta supone el 18/11/2026): **pendiente de confirmar**.
- Firmas de aprobación: **pendiente**.
- Revisión humana de ≥ 20 errores (`reports/error_analysis.csv`, columnas `comentario_equipo` y
  `revisado_por`): **pendiente**.
