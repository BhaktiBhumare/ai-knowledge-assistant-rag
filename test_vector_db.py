
from database import get_connection, save_chunks, search_chunks

# Create a sample 384-dimensional embedding
sample_embedding = [0.1] * 384

try:
    # Save one test chunk
    save_chunks(
        "test_document",
        ["PostgreSQL with pgvector stores text and embeddings."],
        [sample_embedding],
    )
    print("Test chunk saved successfully!")

    # Search for the test chunk
    results = search_chunks(sample_embedding, top_k=1)

    print("\nSearch results:")
    for document_name, chunk_text, similarity in results:
        print("Document:", document_name)
        print("Chunk:", chunk_text)
        print("Similarity:", round(float(similarity), 4))

except Exception as error:
    print("Database test failed:", error)
