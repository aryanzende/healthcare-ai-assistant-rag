# Healthcare AI Assistant

## Objective
Healthcare AI Assistant is a local-first, hackathon-ready RAG system for healthcare policy and operations Q&A. It answers only from ingested PDF documents and routes appointment-booking intents to a mock appointment tool.

## Features
- FastAPI backend with production-style API structure
- PDF ingestion pipeline with ChromaDB storage
- Sentence-transformers embeddings (`all-MiniLM-L6-v2`)
- Groq LLM answer generation with safety-constrained prompting
- Agentic routing:
  - `appointment_tool` for scheduling-related queries
  - `rag` for document-grounded answers
- Confidence scoring (`low`, `medium`, `high`) from retrieval strength
- Streamlit frontend for end-to-end demo
- Basic tests for health, validation, and routing
- Docker + docker-compose support

## Project Structure
```text
healthcare-ai-assistant/
+-- app/
¦   +-- __init__.py
¦   +-- main.py
¦   +-- config.py
¦   +-- embeddings.py
¦   +-- rag.py
¦   +-- llm.py
¦   +-- agent.py
¦   +-- schemas.py
¦   +-- logger.py
¦   +-- utils.py
+-- data/
+-- vector_store/
+-- tests/
+-- streamlit_app.py
+-- requirements.txt
+-- Dockerfile
+-- docker-compose.yml
+-- .env.example
+-- .gitignore
+-- README.md
```

## Architecture Flow
1. Ingest PDFs from `data/`
2. Extract + paragraph/sentence-aware chunk text
3. Generate embeddings via sentence-transformers
4. Store vectors + metadata in ChromaDB
5. `/ask` request enters agent router:
   - Appointment intent -> mock slot tool
   - Otherwise -> retrieve chunks -> Groq answer
6. API returns answer, sources, confidence, and route

## Tech Stack
- Backend: FastAPI, Uvicorn
- LLM: Groq (`llama-3.3-70b-versatile` default)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector DB: ChromaDB (persistent local store)
- PDF parsing: `pypdf`
- Frontend: Streamlit
- Testing: Pytest

## RAG Pipeline
- PDF extraction with error handling
- Paragraph/sentence-aware chunking with overlap
- Metadata stored per chunk:
  - `source` filename
  - `chunk_index`
  - `document_type` inferred from filename
- Similarity filtering by distance threshold
- Weak/irrelevant retrieval returns no context so assistant responds:
  - `I could not find this information in the provided documents.`

## Agentic Workflow
- Appointment keywords trigger routing to `appointment_tool`
- Extracts:
  - Department: cardiology, dermatology, general medicine, orthopedics, pediatrics
  - Day: Monday to Saturday
- Returns mock/demo slot availability with clear disclaimer

## Prompting Strategy
The LLM prompt enforces:
- Answer from retrieved context only
- No hallucination/no outside facts
- Explicit unavailable-information fallback
- No diagnosis and no unsafe medical advice
- Recommend qualified healthcare professional for clinical decisions
- Short, professional, source-aware response style

## Setup (Windows PowerShell)
```powershell
cd D:\RAG_Pipline\healthcare-ai-assistant
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Then edit .env and add your GROQ_API_KEY
```

## Run FastAPI Backend
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Run Streamlit Frontend
```powershell
streamlit run streamlit_app.py
```

## API Endpoints
- `GET /health`
- `POST /ingest`
- `POST /ask`

## Example curl Requests
### Health
```bash
curl http://127.0.0.1:8000/health
```

### Ingest
```bash
curl -X POST http://127.0.0.1:8000/ingest
```

### Ask
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Can patients request medication refills through telehealth?"}'
```

## Sample Questions
- Can patients request medication refills through telehealth?
- What documents are needed for insurance eligibility verification?
- Can healthcare staff share patient records with unauthorized people?
- What should patients do after hospital discharge?
- Can I book a cardiology appointment for Monday?
- What is the hospital cafeteria menu today?

## Tests
```powershell
pytest -q
```

## Docker
```powershell
docker compose up --build
```

## Limitations
- Appointment scheduling is mock/demo only
- Quality depends on uploaded document coverage
- No EHR/hospital integration
- No multilingual optimization yet

## Future Improvements
- Add authentication and API rate limiting
- Add citations with page numbers
- Add reranking for stronger retrieval
- Add structured tool integrations for real hospital systems
- Add observability dashboards and tracing

## Healthcare Safety and Privacy Note
- This assistant is informational and document-grounded only.
- It does not replace medical professionals.
- It should not be used for emergency or diagnostic decisions.
- Do not ingest real patient-identifiable data in demo environments.
