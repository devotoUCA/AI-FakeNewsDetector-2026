# 02 · Iteración 3 (cierre): qué se verificó, qué se decidió y por qué

## Verificado al empezar

- 7 commits de la iteración 2 sobre `dev`; árbol limpio; 16 entradas en el log.
- Las cifras de partida coinciden con `reports/`. Ningún error comentado por el equipo todavía.

## Hallazgos que cambian lo reportado antes

1. **El atajo de oraciones pegadas era mucho más fuerte de lo reportado.** AUC 0,94 como rasgo único;
   la prueba de invarianza en ambos sentidos da 7,65 % de cambio en el modelo final (la iteración 2
   reportó 1,25 %).
2. **La "mejora de generalización" de la iteración 2 fue sobre todo un corrimiento de sesgo**
   (`reports/glue_diagnostic_v3.json`). Primero la atribuí a una transferencia directa del atajo; el
   diagnóstico lo descartó (solo el 5,9 % de las falsas de WELFake lo tiene).
3. **El modelo sí tiene señal de fuente afuera** (AUC 0,714 en WELFake holdout), pero **ninguna de
   veracidad** (AUC 0,490 en FA-KES).
4. **El log estaba incompleto en dos puntos, no uno**: además de LIAR, la verificación `make all` de la
   iteración 1 evaluó test y LIAR en una copia limpia. Ambos quedaron como entradas retroactivas.

## Decisiones

| Decisión | Motivo |
|---|---|
| FA-KES como holdout virgen | Descargable sin login, CC BY 4.0, sin solapamiento relevante, y su etiqueta no está confundida con la fuente |
| La dirección de etiquetas de FA-KES se tomó de la descripción y de la leyenda del archivo | No se pudo verificar por cruce; se declara |
| **Se conserva el clasificador de la iteración 2** | Regla de aceptación del prompt (sección C): los candidatos `v2_1` degradan más de 0,02 en subject, tiempo y WELFake dev. **Pero ese clasificador no cumple la invarianza.** La regla se escribió sin conocer esto. |
| Alternativa no adoptada: `lrchar_v21_C10` | Cumple la invarianza y es más limpio, con peor panel. Cambiarlo es `final_v3.candidate` en `config.yaml`, más otra mirada a los reservados. |
| Compuerta con detector combinado | Mayor detección a 97 % de retención en la mitad de evaluación |
| Fuera de dominio, `p_false = None` y el valor queda en `p_false_reference` | Sin veredicto, sin perder la información para diagnóstico |
| Probabilidad pública acotada a [0,001; 0,999] | Sin 0 ni 1 literales; se muestra "< 1 %" y "> 99 %" |
| Compuerta guardada como JSON comprimido (`models/domain_gate_v3.json.gz`, 180 KB) | Versionable; permite que el quickstart no descargue WELFake |
| Invarianza incumplida como `xfail` estricto | El incumplimiento queda visible en pytest, sin relajar el umbral |
| Embeddings: MiniLM en un entorno aparte, sobre texto `v2_1` | Sin dependencias en el núcleo; `requirements-extra.txt` |
| Recomputación técnica tras un bug de pandas | El análisis de errores falló después de guardar las métricas. Se corrigió y se recalculó solo ese CSV, registrado en el log como `reacceso: true`. |
| Informe histórico movido a `docs/bitacora_iteraciones.md` | El informe final queda como documento de cierre |

## Errores propios declarados

- Iteración 2: medí el atajo de pegado en un solo sentido y describí como mejora de generalización lo
  que fue un corrimiento de sesgo.
- Iteración 2: no revisé el ECE in-domain antes de la evaluación final.
- Iteración 3: primero afirmé que la mejora "era el atajo transferido" y un diagnóstico lo refutó.

## Resultado

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
