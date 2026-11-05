# Guion de defensa (10–12 diapositivas)

**Narrativa:** medimos, detectamos que era engañoso, lo mitigamos y declaramos dónde vale.
Duración sugerida: unos 12 minutos más preguntas. Las cifras salen de `reports/`.

| # | Diapositiva | Mensaje clave | Figura o fuente |
|---|---|---|---|
| 1 | Problema y alcance | Estimar, a partir del texto, si una noticia en inglés se parece a las falsas. **No verifica hechos.** | acta |
| 2 | Un 99 % que no creímos | El acta decía: desconfiar de más de 98 %. El modelo dio 99 %. | informe §3.1 |
| 3 | Datos | ISOT: todas las reales son de Reuters. La clase coincide con la fuente. | `docs/data_card.md` |
| 4 | Cazamos atajos | Un solo rasgo clasifica casi todo: `subject`, el dateline, la oración pegada, las comillas. | `shortcut_audit_v2.png`, `format_shortcuts_v3.png` |
| 5 | Evaluamos robustez, no solo in-domain | Panel: cambio de subject, cambio temporal, externos, invarianza. Test protegido y log de accesos. | `panel_v3.csv`, `test_access_log.jsonl` |
| 6 | **La imagen central** | 99 % adentro, nivel trivial afuera. | `resumen_generalizacion.png` |
| 7 | La prueba decisiva: FA-KES | Holdout virgen, misma fuente en ambas clases, etiqueta de veracidad: AUC de azar. El modelo mide fuente, no verdad. | informe §4 |
| 8 | Lo que no funcionó | Limpiar más empeora, `said` predice también afuera, embeddings no ayudan, y nuestra "mejora" de la iteración 2 era un corrimiento de sesgo. | informe §6 |
| 9 | Mitigación: dominio de validez | Dentro de dominio, probabilidad calibrada. Fuera, sin veredicto. Con sus límites: solo reconoce lo que vio. | informe §5 |
| 10 | Demo | En vivo: nota tipo ISOT (dentro), afirmación corta (fuera), texto en español (fuera). | `python -m fakenews.tasks demo` |
| 11 | Errores y riesgos | Revisión del equipo sobre `error_analysis_v3.csv`. Nunca como único criterio de moderación. | informe §7–8 |
| 12 | Cierre | Cumplimos el acta y cerramos la alerta con evidencia. El arreglo de fondo son los datos. | informe §9–11 |

## Preguntas probables y respuestas cortas

- **"¿98,9 % no es demasiado?"** Sí, y lo probamos: no hay filtración, pero el modelo distingue
  fuentes, no verdad.
- **"Entonces, ¿el modelo no sirve?"** Sirve para lo que mide: reconocer el estilo de las fuentes de
  ISOT. Como detector de noticias falsas en general, no. Decirlo con evidencia es el resultado.
- **"¿Por qué conservaron un modelo que no cumple la invarianza?"** Por la regla que fijamos antes de
  conocer ese dato. Lo declaramos, con la alternativa más limpia y su costo.
- **"¿Por qué no BERT?"** Probamos embeddings congelados y dieron peor. El límite está en la etiqueta
  de los datos, no en la capacidad del modelo.
- **"¿Cómo saben que no miraron el test?"** Lo miramos más de una vez y está declarado en un log
  verificable. Por eso FA-KES, tocado una sola vez, es la cifra más fuerte.
- **"¿Qué significa estar dentro de dominio?"** Que el texto se parece a ISOT. No que la respuesta sea
  correcta.
