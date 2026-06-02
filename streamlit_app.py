from __future__ import annotations

import requests
import streamlit as st


# For local backend testing:
#API_BASE_URL = "http://127.0.0.1:8000"

# For deployed Railway backend:
API_BASE_URL = "https://healthcare-ai-assistant-rag-production.up.railway.app"


st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🏥",
    layout="wide",
)

st.title("Healthcare AI Assistant")
st.caption("Document-grounded healthcare policy assistant")


if "ingest_result" not in st.session_state:
    st.session_state.ingest_result = None

if "ingest_error" not in st.session_state:
    st.session_state.ingest_error = None


with st.sidebar:
    st.header("Controls")

    st.caption(f"Backend: {API_BASE_URL}")

    if st.button("API Health Check"):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=10)
            response.raise_for_status()
            st.success("API is healthy")
            st.json(response.json())

        except Exception as exc:
            st.error(f"Health check failed: {exc}")

    st.subheader("POST /ingest")

    col_exec, col_clear = st.columns(2)

    execute_ingest = col_exec.button("Execute", use_container_width=True)
    clear_ingest = col_clear.button("Clear", use_container_width=True)

    if execute_ingest:
        try:
            with st.spinner("Ingesting documents..."):
                response = requests.post(f"{API_BASE_URL}/ingest", timeout=120)

            st.session_state.ingest_result = response.json()
            st.session_state.ingest_error = None if response.ok else "Ingestion failed"

        except Exception as exc:
            st.session_state.ingest_result = None
            st.session_state.ingest_error = f"Ingestion error: {exc}"

    if clear_ingest:
        st.session_state.ingest_result = None
        st.session_state.ingest_error = None

    if st.session_state.ingest_error:
        st.error(st.session_state.ingest_error)

    if st.session_state.ingest_result is not None:
        st.json(st.session_state.ingest_result)

    st.markdown("---")
    st.subheader("Project Stack")
    st.markdown("- FastAPI backend")
    st.markdown("- Groq LLM")
    st.markdown("- ChromaDB vector store")
    st.markdown("- sentence-transformers embeddings")
    st.markdown("- LangChain appointment tool")
    st.markdown("- Streamlit frontend")


sample_questions = [
    "Can patients request medication refills through telehealth?",
    "What are the rules for medication refill requests?",
    "When can a medication refill be denied?",
    "What documents are needed for insurance eligibility verification?",
    "Can healthcare staff share patient records with unauthorized people?",
    "What privacy rules should be followed for patient information?",
    "What should patients do after hospital discharge?",
    "When should patients contact the hospital after discharge?",
    "What are the appointment scheduling rules?",
    "Can patients reschedule or cancel an appointment?",
    "What should patients bring for their appointment?",
    "Are telehealth visits allowed for emergency medical conditions?",
    "What types of care are suitable for telehealth consultation?",
    "Can patient data be used in AI testing or demos?",
    "What should be done if a patient has breathing difficulty?",
    "Can I book a cardiology appointment for Monday?",
    "Is there any dermatology slot available on Friday?",
    "Can I schedule an orthopedics appointment for Tuesday?",
    "Can I book a pediatrics appointment for Saturday?",
    "Is a general medicine doctor available on Wednesday?",
    "What is the hospital cafeteria menu today?",
    "What is the doctor's personal phone number?",
    "What is the hospital Wi-Fi password?",
    "What is the CEO's salary?",
    "Can you tell me my blood test result?",
    "Can you diagnose my chest pain?",
    "Should I stop taking my medicine immediately?",
    "Can you prescribe antibiotics for me?",
]


st.subheader("Ask a Question")

selected_sample = st.selectbox(
    "Example questions",
    ["Select an example question"] + sample_questions,
)

default_question = "" if selected_sample == "Select an example question" else selected_sample

question = st.text_input(
    "Your question",
    value=default_question,
)


if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Generating answer..."):
                response = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={"question": question},
                    timeout=120,
                )

            data = response.json()

            if response.ok:
                st.subheader("Answer")
                st.write(data.get("answer", ""))

                col1, col2 = st.columns(2)
                col1.metric("Confidence", data.get("confidence", "unknown"))
                col2.metric("Route", data.get("route", "unknown"))

                sources = data.get("sources", [])
                st.subheader("Sources")

                if not sources:
                    st.info("No source chunks returned.")

                else:
                    for index, source in enumerate(sources, start=1):
                        title = (
                            f"{index}. {source.get('document', 'unknown')} "
                            f"| chunk {source.get('chunk_index', 'N/A')}"
                        )

                        with st.expander(title):
                            st.write(source.get("chunk", ""))

            else:
                st.error(data.get("detail", "Request failed"))

        except Exception as exc:
            st.error(f"Request failed: {exc}")


st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Built with ❤️ by <b>Aryan Zende</b>"
    "</p>",
    unsafe_allow_html=True,
)
