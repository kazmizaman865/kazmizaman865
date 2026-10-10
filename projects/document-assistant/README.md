# Grounded Document Assistant

A FastAPI service that answers questions by retrieving passages from local Markdown documents. Answers include filenames, chunk identifiers, and relevance scores. Questions without lexical support receive an explicit abstention.

This is an **extractive retrieval baseline for a RAG workflow**, not an LLM integration. It runs without an API key and never sends documents to an external provider.

## Run

From this directory, with Python 3.12:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -v
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"What is data leakage?","top_k":2}'
```

Interactive API documentation is available at `/docs` on the running server.

## Design

Markdown paragraphs → TF-IDF unigram/bigram index → cosine similarity → top passages with citations. The index loads once on startup. Add `.md` files under `documents/` and restart to rebuild it. The five example paragraphs are sample educational content, not a production knowledge base.

## Limits and evaluation

Tests cover cited retrieval, unsupported questions, invalid inputs, and an empty corpus. The 0.1 similarity cutoff is a heuristic, not a calibrated confidence score. Lexical retrieval can miss paraphrases and cannot resolve contradictions. Long paragraphs are not subdivided. There is no authentication, persistence, or load testing; keep the default loopback binding for local use. A next experiment would compare embedding retrieval against a labeled question set before adding generation.
