"""Local document search and extractive answers with source citations."""
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentIndex:
    def __init__(self, directory: Path):
        self.chunks = []
        for path in sorted(directory.glob('*.md')):
            paragraphs = re.split(r'\n\s*\n', path.read_text(encoding='utf-8'))
            for number, paragraph in enumerate(paragraphs, 1):
                if paragraph.strip() and not paragraph.strip().startswith('#'):
                    self.chunks.append({'source': path.name, 'chunk': number,
                                        'text': paragraph.strip()})
        if not self.chunks:
            raise ValueError('The document directory must contain nonempty Markdown paragraphs.')
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(c['text'] for c in self.chunks)

    def answer(self, question: str, top_k: int = 3):
        if not question.strip():
            raise ValueError('Question cannot be blank.')
        if not 1 <= top_k <= 5:
            raise ValueError('top_k must be between 1 and 5.')
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        selected = [int(i) for i in scores.argsort()[::-1][:top_k] if scores[i] >= 0.1]
        sources = [{**self.chunks[i], 'score': round(float(scores[i]), 4)} for i in selected]
        return {'answer': '\n\n'.join(c['text'] for c in sources) if sources else
                'No supported answer was found in the supplied documents.',
                'sources': sources, 'method': 'extractive_tfidf'}


index = DocumentIndex(Path(__file__).parent / 'documents')
app = FastAPI(title='Grounded Document Assistant', version='1.0.0')


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=5)


@app.get('/health')
def health():
    return {'status': 'ok', 'chunks': len(index.chunks)}


@app.post('/ask')
def ask(body: Question):
    try:
        return index.answer(body.question, body.top_k)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
