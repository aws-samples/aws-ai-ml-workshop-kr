import streamlit as st
import rag_lib as glib

st.set_page_config(page_title="Retrieval-Augmented Generation")
st.title("Retrieval-Augmented Generation")

if 'vector_index' not in st.session_state:
    with st.spinner("Indexing document..."):
        pdf_path = "../../../dataset/resources/Amazon-com-Inc-2023-Shareholder-Letter.pdf"
        texts = glib.load_pdf(pdf_path)
        chunks = glib.create_text_splitter(texts)
        embeddings = [glib.create_embeddings(chunk) for chunk in chunks]
        st.session_state.documents = chunks
        st.session_state.vector_index = glib.create_vector_index(embeddings)

input_text = st.text_area("Input text", label_visibility="collapsed")
go_button = st.button("Go", type="primary")

if go_button:
    with st.spinner("Working..."):
        response_contents, search_results = glib.generate_rag_response(
            index=st.session_state.vector_index,
            question=input_text,
            documents=st.session_state.documents
        )

        st.write(response_contents)
        with st.expander("Retrieval Results (Top 4):"):
            for result in search_results:
                st.write(f"- {result}\n\n")
