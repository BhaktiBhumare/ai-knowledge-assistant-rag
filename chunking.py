
def split_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks with a small overlap.

    chunk_size: maximum characters per chunk
    overlap: characters shared between adjacent chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


if __name__ == "__main__":
    sample_text = """
    Retrieval-Augmented Generation, or RAG, combines document retrieval
    with a language model. First, the system extracts information from
    documents. Then, it divides the text into chunks. These chunks can
    later be converted into embeddings and searched for relevant content.
    """

    chunks = split_text(sample_text, chunk_size=100, overlap=20)

    for index, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {index}:\n{chunk}")
