
import os

import psycopg
from pgvector.psycopg import register_vector


# --------------------------------------------------
# 1. PostgreSQL connection settings
# --------------------------------------------------

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "dbname": "rag_knowledge_db",
    "user": "postgres",
    "password": os.getenv("PGPASSWORD", ""),
}


# --------------------------------------------------
# 2. Connect to PostgreSQL
# --------------------------------------------------

def get_connection():
    conn = psycopg.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


# --------------------------------------------------
# 3. Save document chunks and embeddings
# --------------------------------------------------

def save_chunks(document_name, chunks, embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks must match the number of embeddings."
        )

    with get_connection() as conn:
        with conn.cursor() as cur:

            # Remove the previous version of this document
            cur.execute(
                """
                DELETE FROM document_chunks
                WHERE document_name = %s
                """,
                (document_name,),
            )

            # Save each chunk and its embedding
            for chunk, embedding in zip(chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO document_chunks
                        (document_name, chunk_text, embedding)
                    VALUES (%s, %s, %s)
                    """,
                    (document_name, chunk, embedding.tolist()),
                )


# --------------------------------------------------
# 4. Search PostgreSQL for similar chunks
# --------------------------------------------------

def search_chunks(
    question_embedding,
    top_k=3,
    document_name=None
):
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    vector_value = question_embedding.tolist()

    with get_connection() as conn:
        with conn.cursor() as cur:

            if document_name is not None:
                cur.execute(
                    """
                    SELECT
                        document_name,
                        chunk_text,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM document_chunks
                    WHERE embedding IS NOT NULL
                      AND document_name = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        vector_value,
                        document_name,
                        vector_value,
                        top_k,
                    ),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        document_name,
                        chunk_text,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM document_chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector_value, vector_value, top_k),
                )

            rows = cur.fetchall()

    return [
        {
            "document_name": row[0],
            "chunk": row[1],
            "score": float(row[2]),
        }
        for row in rows
    ]
