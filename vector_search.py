
import numpy as np
from embeddings import load_embedding_model, generate_embeddings


def search_similar_chunks(
    question,
    chunks,
    chunk_embeddings,
    model,
    top_k=3
):
    """
    Find the most relevant text chunks for a question
    using cosine similarity.
    """

    if not chunks:
        return []

    if len(chunks) != len(chunk_embeddings):
        raise ValueError(
            "The number of chunks must match the number of embeddings."
        )

    if top_k <= 0:
        return []

    # Convert the question into a normalized embedding vector
    question_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Since both sides are normalized, dot product equals cosine similarity
    similarities = np.dot(
        chunk_embeddings,
        question_embedding
    )

    # Sort chunk indices by similarity, highest first
    ranked_indices = np.argsort(similarities)[::-1]

    # Return the top matching chunks
    results = []

    for index in ranked_indices[:min(top_k, len(chunks))]:
        results.append({
            "chunk": chunks[index],
            "score": float(similarities[index])
        })

    return results


def retrieve_relevant_chunks(
    question,
    chunks,
    chunk_embeddings,
    model,
    top_k=3
):
    """
    Reusable retrieval function for the Streamlit application.
    """

    if not question or not question.strip():
        return []

    return search_similar_chunks(
        question=question,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        model=model,
        top_k=top_k
    )


if __name__ == "__main__":
    sample_chunks = [
        "RAG retrieves relevant information from documents and uses it to answer questions.",
        "Machine learning enables computers to learn patterns from data.",
        "Embeddings convert text into numerical vectors for semantic search."
    ]

    print("Loading embedding model...")
    model = load_embedding_model()

    print("\nGenerating chunk embeddings...")
    chunk_embeddings = generate_embeddings(
        sample_chunks,
        model
    )

    question = "How does RAG find information to answer a question?"

    print(f"\nQuestion: {question}")
    print("\nMost relevant chunks:")

    results = retrieve_relevant_chunks(
        question=question,
        chunks=sample_chunks,
        chunk_embeddings=chunk_embeddings,
        model=model,
        top_k=2
    )

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Text: {result['chunk']}")
