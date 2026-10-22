
import streamlit as st
import pymupdf

from chunking import split_text
from embeddings import load_embedding_model, generate_embeddings
from vector_search import retrieve_relevant_chunks
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
    "The assistant retrieves relevant passages and generates answers "
    "using a language model."
)


# --------------------------------------------------
# 2. Load the embedding model only once
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
        # Step 1: Extract text from the PDF
        pdf_bytes = uploaded_file.getvalue()

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
                "No selectable text found in this PDF. "
                "Scanned PDFs may require OCR."
            )

        else:

            st.success(
                f"PDF processed successfully! Pages: {len(page_texts)}"
            )

            # ------------------------------------------
            # 4. Split extracted text into chunks
            # ------------------------------------------

            chunks = split_text(
                extracted_text,
                chunk_size=500,
                overlap=50
            )

            st.subheader("Document Information")
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
            # 5. Generate embeddings
            # ------------------------------------------

            with st.spinner(
                "Generating embeddings for document chunks..."
            ):

                model = get_embedding_model()

                chunk_embeddings = generate_embeddings(
                    chunks,
                    model
                )

            st.success(
                "Document embeddings generated successfully!"
            )

            # ------------------------------------------
            # 6. Ask a question
            # ------------------------------------------

            st.subheader("Ask a Question")

            question = st.text_input(
                "Enter your question about the uploaded PDF",
                placeholder="What is this document about?",
                key="document_question"
            )

            # Handle one-chunk documents separately
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
                    value=min(3, len(chunks))
                )

            # ------------------------------------------
            # 7. Retrieve context and generate answer
            # ------------------------------------------

            if st.button("Get Answer", type="primary"):

                if not question.strip():

                    st.warning("Please enter a question first.")

                else:

                    with st.spinner(
                        "Searching the document for relevant information..."
                    ):

                        results = retrieve_relevant_chunks(
                            question=question,
                            chunks=chunks,
                            chunk_embeddings=chunk_embeddings,
                            model=model,
                            top_k=top_k
                        )

                    if not results:

                        st.info(
                            "No relevant chunks were found in the document."
                        )

                    else:

                        # Combine retrieved chunks into context
                        context = "\n\n".join(
                            result["chunk"] for result in results
                        )

                        # Generate a natural-language answer using Ollama
                        with st.spinner(
                            "Generating an answer with the language model..."
                        ):

                            answer = generate_answer(
                                question=question,
                                context=context
                            )

                        # ----------------------------------
                        # 8. Display the generated answer
                        # ----------------------------------

                        st.subheader("Generated Answer")
                        st.write(answer)

                        # ----------------------------------
                        # 9. Display retrieved source passages
                        # ----------------------------------

                        with st.expander("View retrieved source passages"):

                            for rank, result in enumerate(
                                results,
                                start=1
                            ):

                                st.markdown(f"**Source {rank}**")

                                st.write(
                                    f"Similarity score: "
                                    f"{result['score']:.4f}"
                                )

                                st.write(result["chunk"])
                                st.divider()

    except Exception as error:

        st.error(
            f"Could not process this PDF: {error}"
        )
