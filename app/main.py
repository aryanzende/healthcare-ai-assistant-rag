from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agent import handle_appointment_question, is_appointment_question
from app.llm import generate_answer
from app.rag import ingest_documents, retrieve_context

app = FastAPI(
    title="Healthcare AI Assistant",
    description="Healthcare AI Assistant using RAG, Groq LLM, and Pinecone",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Welcome to Healthcare AI Assistant", "docs": "Go to /docs to test the API"}


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Healthcare AI Assistant is running"}


@app.post("/ingest")
def ingest():
    try:
        return ingest_documents()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error during ingestion.") from exc


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        if is_appointment_question(question):
            return handle_appointment_question(question)

        context_chunks = retrieve_context(question)
        return generate_answer(question, context_chunks)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error while answering question.") from exc
