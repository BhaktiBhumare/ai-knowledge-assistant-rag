
import hashlib

import streamlit as st
import pymupdf

from chunking import split_text
from embeddings import load_embedding_model, generate_embeddings
from database import save_chunks, search_chunks
from llm import generate_answer


# --------------------------------------------------
# 1. Configure the Streamlit application
# --------------------------------------------------

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Knowledge Assistant")
st.write(
    "Upload a PDF and ask questions. "
    "The assistant retrieves relevant passages from PostgreSQL "
    "and generates answers using Ollama."
)


# --------------------------------------------------
# 2. Load the embedding model once
# --------------------------------------------------

@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


# --------------------------------------------------
# 3. Upload a PDF document
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"]
)


if uploaded_file is not None:

    try:
        pdf_bytes = uploaded_file.getvalue()

        # Identify the document by its actual content.
        document_id = hashlib.sha256(pdf_bytes).hexdigest()
        document_name = uploaded_file.name

        # ------------------------------------------
        # 4. Extract PDF text
        # ------------------------------------------

        with pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as pdf:

            page_texts = [
                page.get_text().strip()
                for page in pdf
            ]

        extracted_text = "\n\n".join(
            text for text in page_texts if text
        )

        if not extracted_text.strip():

            st.warning(
                "No selectable text was found in this PDF. "
                "Scanned PDFs may require OCR."
            )

        else:

            st.success(
                f"PDF processed successfully! Pages: {len(page_texts)}"
            )

            # ------------------------------------------
            # 5. Split text into chunks
            # ------------------------------------------

            chunks = split_text(
                extracted_text,
                chunk_size=500,
                overlap=50
            )

            st.subheader("Document Information")
            st.write(f"**Document:** {document_name}")
            st.write(f"**Total chunks:** {len(chunks)}")

            with st.expander("View extracted text"):
                st.text_area(
                    "PDF content",
                    value=extracted_text,
                    height=250
                )

            with st.expander("View text chunks"):
                for index, chunk in enumerate(chunks, start=1):
                    st.markdown(f"**Chunk {index}**")
                    st.write(chunk)

            # ------------------------------------------
            # 6. Generate embeddings and save to PostgreSQL
            # ------------------------------------------

            model = get_embedding_model()

            indexed_documents = st.session_state.setdefault(
                "indexed_documents", {}
            )

            if document_id not in indexed_documents:

                with st.spinner(
                    "Generating embeddings and saving to PostgreSQL..."
                ):

                    chunk_embeddings = generate_embeddings(
                        chunks,
                        model
                    )

                    # Store the content hash as the database document name.
                    save_chunks(
                        document_id,
                        chunks,
                        chunk_embeddings
                    )

                    indexed_documents[document_id] = True

                st.success(
                    "Document chunks and embeddings saved in PostgreSQL!"
                )

            else:
                st.info("This PDF is already indexed in this session.")

            # ------------------------------------------
            # 7. Ask a question
            # ------------------------------------------

            st.subheader("Ask a Question")

            question = st.text_input(
                "Enter your question about the uploaded PDF",
                placeholder="What is this document about?",
                key=f"question_{document_id}"
            )

            if len(chunks) == 1:
                top_k = 1
                st.caption(
                    "One chunk is available, so the app will retrieve it."
                )
            else:
                top_k = st.slider(
                    "Number of relevant chunks to retrieve",
                    min_value=1,
                    max_value=min(5, len(chunks)),
                    value=min(3, len(chunks)),
                    key=f"top_k_{document_id}"
                )

            # ------------------------------------------
            # 8. Retrieve context from PostgreSQL
            # ------------------------------------------

            if st.button(
                "Get Answer",
                type="primary",
                key=f"answer_{document_id}"
            ):

                if not question.strip():

                    st.warning("Please enter a question first.")

                else:

                    with st.spinner(
                        "Searching PostgreSQL for relevant passages..."
                    ):

                        question_embedding = generate_embeddings(
                            [question],
                            model
                        )[0]

                        results = search_chunks(
                            question_embedding,
                            top_k=top_k,
                            document_name=document_id
                        )

                    if not results:

                        st.info(
                            "No relevant chunks were found in this document."
                        )

                    else:

                        context = "\n\n".join(
                            result["chunk"] for result in results
                        )

                        # ----------------------------------
                        # 9. Generate an answer using Ollama
                        # ----------------------------------

                        with st.spinner(
                            "Generating an answer with the language model..."
                        ):

                            answer = generate_answer(
                                question=question,
                                context=context
                            )

                        st.subheader("Generated Answer")
                        st.write(answer)

                        # ----------------------------------
                        # 10. Display retrieved source passages
                        # ----------------------------------

                        with st.expander(
                            "View retrieved source passages"
                        ):

                            for rank, result in enumerate(
                                results,
                                start=1
                            ):

                                st.markdown(f"**Source {rank}**")

                                st.write(
                                    "Similarity score: "
                                    f"{result['score']:.4f}"
                                )

                                st.write(result["chunk"])
                                st.divider()

    except Exception as error:
        st.error(f"Could not process this PDF: {error}")
