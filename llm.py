
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"


def generate_answer(question, context):
    """
    Generate an answer using the question and
    relevant text retrieved from the PDF.
    """

    if not question or not question.strip():
        return "Please enter a question."

    if not context or not context.strip():
        return "No relevant context was found in the document."

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using only the context provided below.

If the context does not contain enough information,
clearly say that the answer is not available in the document.
Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )

        response.raise_for_status()

        result = response.json()

        return result.get(
            "response",
            "The model did not return an answer."
        ).strip()

    except requests.exceptions.ConnectionError:
        return (
            "Could not connect to Ollama. "
            "Make sure Ollama is running on your computer."
        )

    except requests.exceptions.Timeout:
        return (
            "The model took too long to respond. "
            "Please try again."
        )

    except requests.exceptions.RequestException as error:
        return f"Error communicating with Ollama: {error}"


if __name__ == "__main__":
    sample_question = "What is RAG?"
    sample_context = (
        "Retrieval-Augmented Generation combines "
        "information retrieval with language models "
        "to generate context-aware answers."
    )

    print("Generating answer...")

    answer = generate_answer(
        sample_question,
        sample_context
    )

    print("\nAnswer:")
    print(answer)
