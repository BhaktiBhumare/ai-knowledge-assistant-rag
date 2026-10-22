
from sentence_transformers import SentenceTransformer


def load_embedding_model():
    """Load a pretrained model that converts text into vectors."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def generate_embeddings(text_chunks, model):
    """Convert a list of text chunks into numerical vectors."""
    embeddings = model.encode(
        text_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings


if __name__ == "__main__":
    sample_chunks = [
        "RAG retrieves relevant information from documents.",
        "Machine learning enables computers to learn from data.",
        "RAG combines document retrieval with a language model."
    ]

    print("Loading embedding model...")
    model = load_embedding_model()

    print("\nGenerating embeddings...")
    vectors = generate_embeddings(sample_chunks, model)

    print("Number of chunks:", len(sample_chunks))
    print("Embedding shape:", vectors.shape)
    print("First vector (first 10 values):")
    print(vectors[0][:10])
