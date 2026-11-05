"""Demo del Detector de Fake News (Streamlit). Ejecutar: `streamlit run app/demo.py`."""

from __future__ import annotations

import streamlit as st

from fakenews.demo_logic import PERMANENT_NOTICE, analyze, example_texts, load_predictor

st.set_page_config(page_title="Detector de Fake News", page_icon="📰", layout="centered")
st.title("Detector de noticias falsas")
st.caption("Proyecto de Inteligencia Artificial · UCA 2026 · clasificador de texto en inglés")
st.warning(PERMANENT_NOTICE)


@st.cache_resource(show_spinner="Cargando modelo...")
def _predictor():
    return load_predictor()


predictor, error = _predictor()
if error:
    st.error(error)
    st.code("make train   # o: python -m fakenews.cli train", language="bash")
    st.stop()

examples = example_texts()
choice = st.selectbox("Ejemplos de prueba (opcional)", ["— Escribir o pegar un texto —"] +
                      [e["nombre"] for e in examples])
default = next((e["texto"] for e in examples if e["nombre"] == choice), "")
text = st.text_area("Pegá el texto de la noticia (título y/o cuerpo, en inglés)", value=default,
                    height=260, max_chars=60000)

if st.button("Analizar", type="primary"):
    r = analyze(predictor, text)
    if not r["en_dominio"]:
        st.subheader(r["titulo_resultado"])
        st.error(r["mensaje_dominio"])
        for w in r["advertencias"]:
            st.warning(w)
        if r["referencia_texto"] is not None:
            with st.expander("Valor de referencia (NO confiable fuera del dominio)"):
                st.write(f"El modelo daría una probabilidad de falsa de {r['referencia_texto']}, pero fuera del "
                         "dominio de validez ese número no tiene respaldo: en esos textos el modelo no supera a "
                         "un clasificador trivial.")
    else:
        st.success("Dentro del dominio de validez: el texto se parece a las noticias de entrenamiento.")
        st.subheader(r["titulo_resultado"])
        c1, c2 = st.columns(2)
        c1.metric("Probabilidad de que sea falsa", r["p_false_texto"])
        c2.metric("Nivel de confianza", r["confianza"])
        st.progress(r["barra"], text="0 % = parecida a las reales · 100 % = parecida a las falsas")
        if r["zona_ambigua"]:
            st.info(r["aviso_zona"])
        for w in r["advertencias"]:
            st.warning(w)
        if r["terminos_falsa"] or r["terminos_real"]:
            st.markdown("**Términos del texto que más pesaron en la decisión**")
            a, b = st.columns(2)
            a.markdown("Empujan hacia *falsa*")
            a.table([{"término": d["termino"], "peso": d["peso"]} for d in r["terminos_falsa"]])
            b.markdown("Empujan hacia *real*")
            b.table([{"término": d["termino"], "peso": d["peso"]} for d in r["terminos_real"]])
        if r["similares"]:
            st.markdown("**Noticias del entrenamiento más parecidas** (similitud coseno; trazabilidad, no prueba)")
            st.table(r["similares"])
    with st.expander("Detalles técnicos"):
        st.write(f"Puntaje de dominio: {r['puntaje_dominio']}")
        st.write(f"Perfil de limpieza: {r['perfil']}")
    st.caption(f"Modelo: {r['version']}")
