# Company Document Chatbot

A RAG (Retrieval-Augmented Generation) system that answers questions based solely on your company documents. No hallucinations — only facts from what you upload.

## Tech Stack

- **LLM**: Cohere Command R+ via OpenRouter
- **Embeddings**: Cohere `embed-multilingual-v3.0` (supports Arabic)
- **Vector Database**: Qdrant (local file-based or self-hosted)
- **Framework**: LangChain + FastAPI + Streamlit

## Project Structure

```
.
├── config.py               # Centralized settings (reads from .env)
├── vector_store.py         # Qdrant client wrapper
├── document_processor.py   # PDF/DOCX/TXT loading and chunking
├── rag_pipeline.py         # RAG chatbot logic
├── app_streamlit.py        # Streamlit UI
├── api_server.py           # FastAPI REST API
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

## Local Setup

**1. Clone and install dependencies**
```bash
git clone <repo-url>
cd company-document-chatbot

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

**2. Configure environment**
```bash
copy .env.example .env   # Windows
# or
cp .env.example .env     # Linux/Mac
```

Open `.env` and fill in your API keys:
```
OPENROUTER_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
```

**3. Run the Streamlit UI**
```bash
streamlit run app_streamlit.py
```

Then open http://localhost:8501, click **Initialize Chatbot**, upload your documents, and start asking questions.

**4. (Optional) Run the FastAPI backend**
```bash
uvicorn api_server:app --reload
```

API docs available at http://localhost:8000/docs

## Docker (Production)

```bash
# Set your API keys in .env first, then:
docker-compose up -d
```

Services:
- Streamlit UI: http://localhost:8501
- FastAPI: http://localhost:8000
- Qdrant: http://localhost:6333

## API Usage

**Chat:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the leave policy?", "include_sources": true}'
```

**Upload documents:**
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "files=@policy.pdf"
```

**Health check:**
```bash
curl http://localhost:8000/health
```

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `cohere/command-r-plus-08-2024` | Model via OpenRouter |
| `EMBEDDING_MODEL` | `embed-multilingual-v3.0` | Cohere embedding model |
| `QDRANT_URL` | `local` | `local` for file-based, or a remote URL |
| `TOP_K_DOCUMENTS` | `5` | Number of chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.1` | Minimum similarity score to include a result |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |


