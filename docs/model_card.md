# Model card — Detector de Fake News

**Versión:** `final_model_v3` (LR de n-gramas de caracteres, limpieza `v2`, calibración isotónica, con
compuerta de dominio). Proyecto académico, UCA 2026. Metadata: `models/metadata_v3.json`.

## Uso previsto

Material didáctico para alfabetización mediática y para la defensa de la materia: mostrar cómo un
modelo con 99 % de accuracy puede no estar midiendo lo que parece.

## Usos NO recomendados

- **Moderar, etiquetar o censurar contenido real.**
- Evaluar medios, periodistas o personas.
- Decidir si una noticia es verdadera. **No verifica hechos.**

## Dominio de validez

Texto en inglés con forma de artículo, parecido a las noticias de ISOT (política de EE.UU. y mundo,
2015–2018). Fuera de ese dominio, el sistema responde "incierta, requiere revisión humana" con la
advertencia `fuera_de_dominio` y no emite probabilidad.

La compuerta **solo reconoce bien lo que vio como negativo**: deja pasar parte de los estilos que no
vio. "Dentro de dominio" significa "se parece a ISOT", no "la respuesta es correcta".

## Datos

Entrenamiento: ISOT (Kaggle, CC BY-NC-SA 4.0). Las reales son todas de Reuters y las falsas de sitios
poco fiables: **la etiqueta es por fuente**. Evaluación externa: WELFake (CC BY 4.0), LIAR (solo
investigación) y FA-KES (CC BY 4.0). Detalle en `docs/data_card.md`.

## Métricas honestas

<!-- BEGIN GENERATED:readme -->
| Conjunto | Accuracy (forzando respuesta) | Trivial | Fuera de dominio | Accuracy sobre lo respondido | Trivial sobre lo mismo |
|---|---|---|---|---|---|
| ISOT test | 0,990 | 0,547 | 2,9 % | 0,990 | 0,561 |
| WELFake holdout | 0,581 | 0,583 | 70,3 % | 0,640 | 0,502 |
| LIAR test (no virgen) | 0,514 | 0,574 | 92,1 % | 0,508 | 0,460 |
| FA-KES (virgen) | 0,531 | 0,530 | 40,8 % | 0,538 | 0,540 |
<!-- END GENERATED:readme -->

- ISOT test y WELFake holdout ya se habían mirado antes; **FA-KES es la única cifra virgen**.
- En FA-KES la AUC es 0,490: **azar**. Es el conjunto donde la etiqueta es de veracidad y la fuente no
  separa las clases.
- Calibración: excelente dentro de ISOT; mala afuera, incluso sobre lo que pasa la compuerta.

## Limitaciones

- Aprende la redacción de las fuentes, no la veracidad.
- **No cumple el criterio de invarianza** en `oraciones_pegadas`: normalizar o pegar oraciones cambia
  el 7,65 % de las predicciones (límite: cota superior de 2 %).
- Solo inglés. Falla con sátira, noticias nuevas y temas fuera del dataset.

## Riesgos éticos

- Un falso positivo puede desacreditar a un medio legítimo; afuera, el error dominante es justamente
  marcar como falsas noticias reales de medios que no son Reuters.
- Un falso negativo da falso respaldo a una noticia falsa.
- Una confianza "alta" puede leerse como certeza: significa parecido con el entrenamiento.
- Sesgo hacia el registro formal de agencia, en contra del registro coloquial.

Más detalle: `docs/informe_resultados.md`.
