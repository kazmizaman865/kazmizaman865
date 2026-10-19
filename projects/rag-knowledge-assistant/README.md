# RAG Knowledge Assistant

A runnable retrieval-augmented question answering API built with FastAPI, TF-IDF retrieval and optional OpenAI generation.

**Status:** starter implementation / portfolio demonstration. No benchmark or production claim is implied.

## Features
- Upload or paste text documents with a local JSON-backed document store
- Chunked retrieval with TF-IDF and cosine similarity
- Source-backed answers with citations to retrieved chunks
- Optional LLM answer generation using `OPENAI_API_KEY`
- FastAPI interactive docs at `/docs`
- Unit tests and Docker configuration

## Quickstart

```bash
cd projects/rag-knowledge-assistant
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs and call `POST /documents` then `POST /ask`.

Example document:
```json
{"title":"Project Notes","text":"Retrieval augmented generation finds relevant document passages before generating an answer."}
```

Example question:
```json
{"question":"What does retrieval augmented generation do?","top_k":3}
```

Without an API key, the service returns relevant excerpts instead of generating unsupported text. For LLM responses, set `OPENAI_API_KEY`; never commit your key.

## Tests
```bash
pytest -q
```

## Docker
```bash
docker build -t rag-assistant .
docker run --rm -p 8000:8000 -v rag_data:/app/data rag-assistant
```

## Production next steps
Authentication, document parsers, vector database (FAISS/Chroma), hybrid search, prompt-injection defenses, rate limiting, evaluation datasets and hosted deployment.
