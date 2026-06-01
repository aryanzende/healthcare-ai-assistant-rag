# Healthcare AI Assistant Architecture

## Architecture Flowchart
```text
                    +------------------------------+
                    | User / Demo UI               |
                    | Streamlit or API Client      |
                    +--------------+---------------+
                                   |
                                   v
                    +------------------------------+
                    | FastAPI Backend              |
                    | /health  /ingest  /ask       |
                    +--------------+---------------+
                                   |
                                   v
                    +------------------------------+
                    | Question Router              |
                    | Appointment or RAG Query     |
                    +--------------+---------------+
                                   |
                     +-------------+-------------+
                     |                           |
                     v                           v
       +---------------------------+   +---------------------------+
       | Mock Appointment Tool     |   | RAG Pipeline              |
       | Department + Day Slots    |   | Retrieve + Generate       |
       +-------------+-------------+   +-------------+-------------+
                     |                               |
                     v                               v
       +---------------------------+   +---------------------------+
       | Appointment Response      |   | ChromaDB Similarity       |
       | route: appointment_tool   |   | + Context Retrieval       |
       +-------------+-------------+   +-------------+-------------+
                                                   |
                                                   v
                                     +---------------------------+
                                     | Groq LLM                  |
                                     | Safety-Grounded Prompt    |
                                     +-------------+-------------+
                                                   |
                                                   v
                                     +---------------------------+
                                     | Final API Response        |
                                     | answer + sources +        |
                                     | confidence + route        |
                                     +---------------------------+

Ingestion path:
POST /ingest -> Read PDFs -> Extract text (pypdf) -> Chunk -> Embed -> Store in ChromaDB
```

## Objective
Build a healthcare Q&A assistant that answers only from provided PDF documents using RAG, with a simple appointment-routing tool for demo purposes.

## Tech Stack
- FastAPI (API)
- Streamlit (UI)
- Groq LLM (answer generation)
- sentence-transformers/all-MiniLM-L6-v2 (embeddings)
- ChromaDB (vector search)
- pypdf (PDF text extraction)
- Docker (portable run)

## System Flow
1. User ingests PDFs via `POST /ingest`.
2. PDFs are parsed, chunked, embedded, and stored in ChromaDB.
3. User asks via `POST /ask`.
4. Router decides:
   - Appointment query -> mock appointment tool (`route: appointment_tool`)
   - Other query -> RAG retrieval + Groq (`route: rag`)
5. API returns `answer`, `sources`, `confidence`, and `route`.

## RAG Pipeline
- Read PDFs from `data/`.
- Extract/clean text with `pypdf`.
- Chunk text with overlap.
- Create embeddings and store with metadata (`source`, `chunk_index`, `document_type`).
- Retrieve top chunks by similarity and filter weak matches.
- Send context + question to Groq with a strict grounded prompt.

## API Endpoints
- `GET /` : welcome
- `GET /health` : health status
- `POST /ingest` : rebuild vector store from PDFs
- `POST /ask` : answer query (router + RAG/tool)

## Agentic Workflow
Appointment intents are detected using keywords. The tool extracts:
- Department: cardiology, dermatology, general medicine, orthopedics, pediatrics
- Day: Monday to Saturday

It returns mock slots with clear demo-only messaging.

## Prompting Strategy
The LLM is instructed to:
- Use only retrieved context
- Never guess/use outside knowledge
- Return fallback exactly when answer is missing:
  `I could not find this information in the provided documents.`
- Avoid diagnosis and unsafe advice
- Recommend a qualified healthcare professional for clinical decisions/emergencies

## Source Citation Strategy
Each response returns source chunks:
- `document`
- `chunk_index`
- `chunk`

This keeps answers auditable and easy to verify.

## Key Decisions
- Keep architecture simple and local-first for hackathon speed.
- Use ChromaDB + MiniLM for lightweight retrieval.
- Use deterministic appointment routing for explainable demo behavior.

## Limitations
- Works only on ingested document coverage.
- No OCR for scanned PDFs.
- Appointment tool is mock-only.
- No real EHR integration or auth.

## Future Improvements
- Add OCR support
- Add page-level citations
- Add retrieval reranking
- Add authentication and PHI safeguards
- Integrate real appointment APIs

## Healthcare Safety & Privacy Note
This project uses synthetic/demo documents and should not be used as a substitute for professional medical care. Do not ingest real patient-identifiable data.
