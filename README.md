# Healthcare AI Assistant Using RAG and LLMs

A healthcare-focused AI assistant that answers questions from healthcare PDF documents using Retrieval-Augmented Generation.

This project includes PDF ingestion, semantic search, Groq LLM integration, source citations, FastAPI APIs, Streamlit UI, LangChain appointment tool workflow, and Docker support.

---

## Objective

Build a healthcare AI assistant that answers only from provided healthcare documents and avoids hallucination when information is not available.

---

## Tech Stack

| Component      | Technology                             |
| -------------- | -------------------------------------- |
| Backend        | FastAPI                                |
| Frontend       | Streamlit                              |
| LLM            | Groq                                   |
| Embeddings     | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB      | ChromaDB                               |
| PDF Reader     | pypdf                                  |
| Agent Workflow | LangChain Tool                         |
| Deployment     | Docker                                 |

---

## Features

* PDF document ingestion
* RAG-based question answering
* Source citations with document name and chunk index
* Unknown answer handling
* Healthcare-safe prompting
* LangChain mock appointment tool
* FastAPI endpoints
* Streamlit demo UI
* Dockerfile and docker-compose support

---

## Architecture Flow

```text
User / Streamlit / Swagger
        ↓
FastAPI Backend
        ↓
Question Router
        ↓
Appointment Question? ── Yes ──> LangChain Mock Appointment Tool
        ↓ No
RAG Pipeline
        ↓
ChromaDB Similarity Search
        ↓
Retrieved PDF Chunks
        ↓
Groq LLM
        ↓
Answer + Sources + Confidence
```

---

## RAG Pipeline

```text
PDFs in /data
   ↓
Text extraction using pypdf
   ↓
Chunking
   ↓
Embedding generation
   ↓
Store in ChromaDB
   ↓
Retrieve relevant chunks
   ↓
Generate grounded answer using Groq
```

---

## Project Structure

```text
healthcare-ai-assistant/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── embeddings.py
│   ├── rag.py
│   ├── llm.py
│   └── agent.py
├── data/
├── screenshots/
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── ARCHITECTURE.md
└── README.md
```

---

## API Endpoints

| Method | Endpoint  | Purpose                                 |
| ------ | --------- | --------------------------------------- |
| GET    | `/health` | Check API status                        |
| POST   | `/ingest` | Ingest PDFs and build vector store      |
| POST   | `/ask`    | Ask healthcare or appointment questions |

---

## Setup

```powershell
git clone <your-repo-link>
cd healthcare-ai-assistant
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

---

## Run FastAPI

```powershell
uvicorn app.main:app
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Run Streamlit

```powershell
streamlit run streamlit_app.py
```

---

## API Examples

### Ingest Documents

```http
POST /ingest
```

Sample response:

```json
{
  "message": "Documents ingested successfully",
  "total_files": 6,
  "total_chunks": 28
}
```

### Ask RAG Question

```json
{
  "question": "Can patients request medication refills through telehealth?"
}
```

Sample response:

```json
{
  "answer": "Yes, if the medication is already prescribed, the patient is stable, and no in-person exam or restricted monitoring is required.",
  "sources": [
    {
      "document": "telehealth_policy.pdf",
      "chunk_index": 2,
      "chunk": "Medication refill requests may be reviewed during telehealth visits..."
    }
  ],
  "confidence": "medium",
  "route": "rag"
}
```

### Ask Appointment Question

```json
{
  "question": "Can I book a cardiology appointment for Monday?"
}
```

Sample response:

```json
{
  "answer": "This is demo availability. Available slots for cardiology on Monday are: 10:00 AM, 3:00 PM.",
  "confidence": "medium",
  "route": "appointment_tool"
}
```

### Unknown Answer

```json
{
  "question": "What is the CEO's salary?"
}
```

Expected response:

```text
I could not find this information in the provided documents.
```

---

## Prompting Strategy

The Groq LLM prompt ensures:

* Answer only from retrieved context
* Do not use outside knowledge
* Do not guess
* Avoid diagnosis and unsafe medical advice
* Return this fallback when information is missing:

```text
I could not find this information in the provided documents.
```

---

## Dataset Details

The project uses healthcare PDFs inside the `data/` folder.

Topics include:

* Telehealth policy
* Medication refill policy
* Appointment policy
* Privacy guidelines
* Insurance FAQ
* Discharge instructions

No real patient data or PHI is used.

---

## Screenshots

### Document Ingestion API

![Document Ingestion API](screenshots/ingest_api.png)

### Ask API Request

![Ask API Request](screenshots/ask_api_request.png)

### Ask API Response with Sources

![Ask API Response](screenshots/ask_api_response.png)

### Streamlit RAG Answer

![Streamlit RAG Answer](screenshots/streamlit_rag_answer.png)

### LangChain Appointment Tool

![Appointment Tool](screenshots/streamlit_appointment_tool.png)

### Unknown Answer Handling

![Unknown Answer](screenshots/streamlit_unknown_answer.png)

---

## Docker

Build image:

```powershell
docker build -t healthcare-ai-assistant .
```

Run container:

```powershell
docker run -p 8000:8000 --env-file .env healthcare-ai-assistant
```

---

## Key Choices

| Item            | Used                                   |
| --------------- | -------------------------------------- |
| LLM             | Groq                                   |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB       | ChromaDB                               |
| Agent Workflow  | LangChain mock appointment tool        |
| Backend         | FastAPI                                |
| UI              | Streamlit                              |

---

## Limitations

* Answers only from provided PDFs
* Mock appointment system only
* No real hospital/EHR integration
* No scanned PDF OCR
* Not for diagnosis or emergency use

---

## Future Improvements

* Add PDF upload from UI
* Add OCR for scanned PDFs
* Add page-level citations
* Add authentication
* Add PHI masking
* Add real appointment API integration
* Deploy on Render

---

## Healthcare Safety Note

This project is for demo purposes only. It does not use real patient data or PHI. It should not be used for diagnosis, treatment, or emergency decisions.
