# Guion de defensa (10–12 diapositivas)

Duración sugerida: unos 12 minutos más preguntas. Todas las cifras salen de `reports/`.

| # | Diapositiva | Mensaje clave | Figura o fuente |
|---|---|---|---|
| 1 | Problema y alcance | Estimar, a partir del texto, si una noticia en inglés se parece a las falsas. Clasificador binario con probabilidad. **No verifica hechos.** | acta |
| 2 | Datos | ISOT/Kaggle (CC BY-NC-SA 4.0): 44 898 → 38 199 tras limpiar y deduplicar. Las reales son todas de Reuters. LIAR como prueba externa. | `docs/data_card.md` |
| 3 | La fuga que encontramos | `(Reuters) -` solo ya acierta el 99,6 %; `subject` y el formato de fecha, el 100 %; también la codificación de comillas. Mostrar la tabla de marcas. | `leakage_audit.json`, notebook §5 |
| 4 | Preparación | Una sola limpieza para train e inferencia; deduplicación exacta y MinHash **antes** de dividir; 70/15/15 sin solapamiento. Test sagrado: 1 acceso por modelo, registrado. | `splits_manifest.json`, `test_access_log.jsonl` |
| 5 | Baseline | Trivial: 0,547 de accuracy y F1 falsa 0. LR TF-IDF: F1 falsa 0,985 en test. Grilla con meseta y brecha train–val de 0,014. | `baseline_logreg_grid.csv` |
| 6 | Comparación | SVM 0,988 (McNemar p = 0,03); ensamble 0,986 (no significativo); MLP = LR → no se adopta; estilo solo 0,92; kNN 0,88. | `f1_by_model_test.png` |
| 7 | Probabilidad y decisión | Calibración isotónica (Brier 0,008). Umbral 0,5 elegido en validación. Zona 0,4–0,6: deriva 0,6 % de los casos, que solo aciertan el 53 %. Confianza alta, media y baja con accuracy de 0,997, 0,79 y 0,53. | `reliability_productive_test.png`, `coverage_test.png` |
| 8 | Experimento controlado | Con marcas F1 0,993, sin marcas 0,985. La caída es chica… porque quedan señales de estilo (`said`, días de la semana frente a `gop`, `just`, `watch`). | `leakage_experiment.png` |
| 9 | **La prueba de fuego: LIAR** | Sin reentrenar, accuracy 0,487, por debajo del trivial (0,560). El modelo aprendió el estilo de las fuentes, no la veracidad. Lo mostramos en vez de esconderlo. | `external_liar.png` |
| 10 | Errores | 61 errores en test; 40 analizados (26 con alta confianza). Ejemplo: Reuters citando tuits → marcado falsa con p ≈ 1. Comentarios del equipo. | `error_analysis.csv` |
| 11 | Demo | En vivo: un ejemplo del test, un texto en español (→ incierta) y un texto corto (→ advertencia). Aviso permanente. | `streamlit run app/demo.py` |
| 12 | Riesgos, límites y próximos pasos | Riesgo de falso respaldo y de censura; nunca usarlo como único criterio. Próximos pasos: etiquetas de veracidad, varias fuentes por clase, validación cruzada por fuente. | informe §12–14 |

## Preguntas probables y respuestas cortas

- **"¿98,9 % no es demasiado?"** Sí, y por eso lo tratamos con sospecha: sacamos las marcas, medimos la
  fuga y lo probamos en LIAR, donde cae al nivel del azar.
- **"¿Por qué no BERT?"** El acta dice "solo si hace falta". El problema no es la capacidad del modelo
  sino los datos: un modelo más potente aprendería el estilo de la fuente todavía mejor.
- **"¿Qué significa la confianza?"** Parecido con el entrenamiento, calibrado en validación. No es
  certeza sobre los hechos.
- **"¿Cómo garantizan que no miraron el test?"** Por el registro append-only y porque todas las
  decisiones están documentadas en `val_results.json` antes del acceso.
