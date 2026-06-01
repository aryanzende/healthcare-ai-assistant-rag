from __future__ import annotations

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.rag import confidence_from_distance


UNKNOWN_ANSWER = "I could not find this information in the provided documents."


def _empty_context_response():
    return {
        "answer": UNKNOWN_ANSWER,
        "sources": [],
        "confidence": "low",
        "route": "rag",
    }


def generate_answer(question: str, context_chunks: list):
    """
    Generate a grounded healthcare answer using Groq LLM.
    The answer must be based only on retrieved document context.
    """

    if not context_chunks:
        return _empty_context_response()

    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY.")

    context_text = ""

    for item in context_chunks:
        context_text += (
            f"\nSource Document: {item['document']}\n"
            f"Chunk Index: {item['chunk_index']}\n"
            f"Content:\n{item['chunk']}\n"
        )

    system_prompt = f"""
You are a healthcare AI assistant for document-based question answering.

Your job is to answer the user's question using ONLY the provided healthcare document context.

Strict rules:
1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not guess or assume missing information.
4. If the answer is not clearly present in the context, reply exactly:
"{UNKNOWN_ANSWER}"
5. Do not provide medical diagnosis.
6. Do not provide unsafe medical advice.
7. For clinical decisions, symptoms, emergencies, or treatment choices, recommend contacting a qualified healthcare professional.
8. Keep the answer short, clear, and professional.
9. Do not mention source names inside the answer unless the context supports it.
"""

    user_prompt = f"""
Context:
{context_text}

User Question:
{question}

Answer:
"""

    try:
        client = Groq(api_key=GROQ_API_KEY)

        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=350,
        )

    except Exception as exc:
        raise RuntimeError(f"Groq API error: {exc}") from exc

    answer = (completion.choices[0].message.content or "").strip()

    if not answer:
        answer = UNKNOWN_ANSWER

    best_distance = min(item["distance"] for item in context_chunks)
    confidence = confidence_from_distance(best_distance)

    return {
        "answer": answer,
        "sources": [
            {
                "document": item["document"],
                "chunk_index": item["chunk_index"],
                "chunk": item["chunk"],
            }
            for item in context_chunks
        ],
        "confidence": confidence,
        "route": "rag",
    }