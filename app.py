
import streamlit as st
import pymupdf
from chunking import split_text

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Knowledge Assistant")
st.write("Upload a PDF to extract its text and split it into chunks.")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    try:
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

        if not extracted_text:
            st.warning(
                "No selectable text found in this PDF. "
                "It may be scanned and require OCR."
            )
        else:
            st.success(
                f"PDF processed successfully! "
                f"Pages: {len(page_texts)}"
            )

            st.subheader("Extracted Text")
            st.text_area(
                "PDF content",
                value=extracted_text,
                height=250
            )

            chunks = split_text(
                extracted_text,
                chunk_size=500,
                overlap=50
            )

            st.subheader("Text Chunks")
            st.write(f"Total chunks created: **{len(chunks)}**")

            for index, chunk in enumerate(chunks, start=1):
                with st.expander(f"Chunk {index}"):
                    st.write(chunk)

    except Exception as error:
        st.error(f"Could not process this PDF: {error}")
