
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"


def ask_ollama(prompt):
    """Generate a response using the local Ollama model."""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 350
            }
        },
        timeout=180
    )

    response.raise_for_status()
    return response.json().get("response", "").strip()


def generate_answer(question, context):
    """
    Answer from the document when supported.
    Otherwise, use general knowledge and clearly label it.
    Avoid unsupported claims and guessing.
    """

    try:
        # STEP 1: Check whether the document supports an answer.
        document_prompt = f"""
You are a careful document question-answering assistant.

Use ONLY the document context below.

Rules:
- Answer only with facts supported by the context.
- Do not add outside information.
- If the answer is not present or cannot be determined,
  return exactly: NOT_FOUND_IN_DOCUMENT
- Do not guess.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        document_answer = ask_ollama(document_prompt)
        normalized = document_answer.strip().upper()

        # STEP 2: If the document does not contain the answer,
        # answer using general knowledge instead.
        if (
            not document_answer
            or "NOT_FOUND_IN_DOCUMENT" in normalized
        ):
            general_prompt = f"""
You are a careful and factual AI assistant.

Answer the question using your general knowledge.

Rules:
- Give only information you have good reason to believe is true.
- Do not invent names, dates, statistics, sources, or details.
- If you are uncertain about an important fact, clearly say so.
- If the answer depends on recent information you cannot verify,
  explain that limitation.
- Use simple English and a relevant example when helpful.
- Do not pretend this answer came from the uploaded document.

QUESTION:
{question}

ANSWER:
"""

            general_answer = ask_ollama(general_prompt)

            if not general_answer:
                return (
                    "I couldn't generate a reliable answer. "
                    "Please try rephrasing your question."
                )

            return (
                "**General knowledge (not found in the document)**\n\n"
                + general_answer
            )

        # STEP 3: Verify the proposed answer against the context.
        verification_prompt = f"""
You are a strict fact-checker.

Compare the proposed answer with the document context.

Return SUPPORTED only if the answer's factual claims are
supported by the context. Otherwise return NOT_SUPPORTED.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

PROPOSED ANSWER:
{document_answer}

Your response must be exactly one of:
SUPPORTED
NOT_SUPPORTED
"""

        verification = ask_ollama(verification_prompt).strip().upper()

        if not verification.startswith("SUPPORTED"):
            # The document answer could not be verified.
            # Fall back to general knowledge rather than showing it.
            general_prompt = f"""
Answer this question using your general knowledge.
Do not guess or invent facts. If uncertain, say so.
Use simple English.

Question: {question}

Answer:
"""

            general_answer = ask_ollama(general_prompt)

            if not general_answer:
                return "I couldn't generate a reliable answer."

            return (
                "**General knowledge (document answer unverified)**\n\n"
                + general_answer
            )

        return (
            "**According to the uploaded document**\n\n"
            + document_answer
        )

    except requests.exceptions.ConnectionError:
        return (
            "Could not connect to Ollama. "
            "Please make sure Ollama is running."
        )

    except requests.exceptions.Timeout:
        return "Ollama took too long to respond. Please try again."

    except requests.exceptions.RequestException as error:
        return f"Error communicating with Ollama: {error}"
