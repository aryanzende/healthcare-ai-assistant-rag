## Screenshots

The following screenshots show the main working parts of the project, including document ingestion, API testing, RAG-based answering, source citations, LangChain appointment tool routing, and unknown-answer handling.

### Document Ingestion API

This shows the `/ingest` endpoint successfully reading healthcare PDFs from the `data/` folder, splitting them into chunks, generating embeddings, and storing them in ChromaDB.

![Document Ingestion API](screenshots/ingest_api.png)

### Ask API Request

This shows the `/ask` endpoint accepting a healthcare question in JSON format.

![Ask API Request](screenshots/ask_api_request.png)

### Ask API Response with Source Citations

This shows the RAG response generated from retrieved healthcare document chunks. The response includes the final answer, source document names, chunk indexes, confidence, and route.

![Ask API Response](screenshots/ask_api_response.png)

### Streamlit RAG Answer

This shows the Streamlit UI answering a document-based healthcare question using the RAG pipeline.

![Streamlit RAG Answer](screenshots/streamlit_rag_answer.png)

### LangChain Appointment Tool Workflow

This shows the basic agentic workflow. Appointment-related questions are routed to the LangChain appointment tool instead of the RAG pipeline.

![Appointment Tool](screenshots/streamlit_appointment_tool.png)

### Unknown Answer Handling

This shows how the assistant handles questions that are not available in the provided documents. Instead of hallucinating, it returns the fallback answer.

![Unknown Answer](screenshots/streamlit_unknown_answer.png)