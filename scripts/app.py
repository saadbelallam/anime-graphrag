"""streamlit run scripts/app.py"""

import streamlit as st

from graphrag.pipeline import ask

st.set_page_config(page_title="anime-graphrag", page_icon="🕸️")
st.title("🕸️ anime-graphrag")
st.caption("Knowledge graph Neo4j + recherche vectorielle, routage par LLM")

EXAMPLES = [
    "Quel studio a produit Fullmetal Alchemist: Brotherhood ?",
    "Un anime mélancolique sur le deuil",
    "J'ai adoré Violet Evergarden, quelque chose d'aussi émouvant du même studio ?",
]

question = st.text_input("Question", placeholder=EXAMPLES[0])

cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example[:38] + "…", use_container_width=True):
        question = example

if question:
    with st.spinner("Recherche…"):
        result = ask(question)

    st.markdown(result.answer)

    with st.expander("Comment la réponse a été construite"):
        st.markdown(f"**Mode** : `{result.mode}`")
        if result.entities:
            st.markdown(f"**Entités** : {', '.join(result.entities)}")
        if result.semantic_query:
            st.markdown(f"**Requête sémantique** : _{result.semantic_query}_")
        if result.anchors:
            st.markdown("**Ancrages** : " + ", ".join(a["title"] for a in result.anchors))
        if result.cypher:
            st.code(result.cypher, language="cypher")
        if result.fallback:
            st.warning(f"Repli sur le vectoriel : {result.fallback}")
