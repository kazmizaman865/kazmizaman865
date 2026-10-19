import os
from fastapi.testclient import TestClient
from app import main


def test_retrieval():
    docs = [{"id": "1", "title": "ML", "text": "Machine learning trains models on data."}]
    assert main.retrieve("How does machine learning train?", docs, 2)[0]["title"] == "ML"


def test_api(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA_FILE", tmp_path / "documents.json")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(main.app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/documents", json={"title": "RAG", "text": "RAG retrieves evidence from documents."}).status_code == 201
    response = client.post("/ask", json={"question": "What retrieves evidence?"})
    assert response.status_code == 200
    assert len(response.json()["sources"]) == 1
