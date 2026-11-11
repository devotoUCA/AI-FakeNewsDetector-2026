# 03 · Cierre: qué falta y cómo cerrarlo

Estado al 11/11/2026. El código, los modelos, el informe y la demo están terminados. Lo que queda no
se puede resolver programando: son verificaciones y decisiones del equipo.

## 1. Revisar y firmar al menos 20 errores (criterio del acta)

`reports/error_analysis_v3.csv` tiene 61 errores del modelo final. Cada fila trae:

- `comentario_borrador`: texto automático (predicción, probabilidad, términos).
- `comentario_equipo`: una **lectura propuesta** del caso, escrita a partir del título, el extracto y
  los términos. Es un punto de partida, no la revisión.
- `revisado_por`: vacío. Acá firma quien revisa.

Un caso cuenta como revisado **solo si tiene comentario y firma**. Para cada caso que tomen:

1. Leer `titulo`, `texto_resumido`, `terminos_hacia_falsa` y `terminos_hacia_real`.
2. Leer la propuesta de `comentario_equipo`. Si están de acuerdo, dejarla; si no, corregirla o
   reescribirla. Varias propuestas dicen que el texto "parece" de tal medio: eso se dedujo del formato
   y conviene confirmarlo o sacarlo.
3. Escribir el nombre de quien revisó en `revisado_por`.

Conviene repartir los 61 entre los cinco y cubrir los cuatro grupos (ISOT y WELFake, FN y FP).

**Si editan con Excel:** guardar como "CSV UTF-8 (delimitado por comas)", sin borrar ni renombrar
columnas. Si Excel abre los acentos rotos, usar Datos → Desde texto/CSV → origen 65001 (UTF-8).

Después:

```bash
python -m fakenews.cli review-status   # o: make review. Cuenta revisados y avisa cuántos faltan
python -m fakenews.cli report          # o: make report. Lleva el estado a las tablas de los documentos
python -m pytest -q -m "not slow"
```

Ninguno de estos comandos carga el modelo ni los conjuntos reservados: no agregan entradas a
`reports/test_access_log.jsonl`. Si alguna vez se vuelve a correr `evaluate-v3`, los comentarios y
las firmas ya cargados se conservan (se cruzan por `id`).

Al llegar a 20 firmados, el criterio pasa solo a "cumplido" en `reports/success_criteria_v3.json`, el
README, el informe (§9) y `docs/campos_completar.md`. Los patrones de §7 del informe salen de la
lectura propuesta: si la revisión los cambia, hay que actualizar ese texto a mano.

## 2. Decisión: qué modelo se defiende

| Opción | A favor | En contra |
|---|---|---|
| Mantener el clasificador de la iteración 2 (**aplicado**) | Mejor puntaje robusto del panel; no requiere otra mirada a los reservados | No cumple la invarianza en `oraciones_pegadas` (queda como `xfail` declarado) |
| `lrchar_v21_C10` | Cumple la invarianza | Peor puntaje robusto; cambiarlo implica reentrenar y reevaluar, con otra mirada a los conjuntos reservados que hay que registrar |

En FA-KES los dos enfoques quedan al nivel del azar, así que el hallazgo central no depende de esta
elección. Si se cambia: `final_v3.candidate` en `config.yaml`, luego `train-v3`, `evaluate-v3` y
`report`.

## 3. Verificaciones que nadie corrió todavía

| Qué | Estado | Cómo |
|---|---|---|
| Linux: instalación y tests sin datos (lo mismo que hace el CI) | **Verificado** el 11/11/2026 (Linux x86_64, Python 3.12): `setup` en ~27 s; 180 pasan, 58 se saltean, 1 `xfail` | — |
| Linux: `quickstart` y demo | No verificado (el entorno de prueba no tenía acceso a Kaggle) | `make setup && make quickstart && make demo` |
| CI en GitHub (Ubuntu y Windows) | No verificado | Pestaña *Actions* del repo, workflow `tests`, después del próximo push a `main` o `dev` |
| Windows: instalación, quickstart y demo | No verificado | Pasos de README §3 y §5; `predict` por stdin: ver README §7 |
| `all` de punta a punta | No se corrió en la iteración 3, a propósito (vuelve a mirar los reservados) | Solo si los docentes lo piden; queda registrado como `reacceso` |

## 4. Datos y acta

- Confirmar las condiciones de uso del ISOT original (University of Victoria); hoy se cita la licencia
  del mirror de Kaggle.
- La dirección de etiquetas de FA-KES se tomó de su descripción y de la leyenda del archivo, sin cruce
  con otro dataset. Si alguien puede verificarla de otra forma, anotarlo en `docs/data_card.md`.
- Roles, fecha real de entrega y firmas del acta: `docs/campos_completar.md`.
