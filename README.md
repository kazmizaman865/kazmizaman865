# Zaman Ali

### AI engineering · Machine learning · Data science

I build AI applications and share practical explanations of data science, AI, and machine learning. My interests include LLM applications, retrieval-augmented generation (RAG), and agentic AI.

[View my portfolio](https://github.com/kazmizaman865/kazmizaman865/tree/main/projects) · [LinkedIn](https://www.linkedin.com/in/sayed-zaman-ali-shah-917a1a222/) · [Email me](mailto:kazmizaman865@gmail.com)

[![Portfolio tests](https://github.com/kazmizaman865/kazmizaman865/actions/workflows/portfolio-tests.yml/badge.svg)](https://github.com/kazmizaman865/kazmizaman865/actions/workflows/portfolio-tests.yml)

## What I work on

- **AI applications:** LLM integrations, RAG pipelines, and machine learning models.
- **Machine learning:** Deep learning, natural language processing, and computer vision.
- **MLOps:** Tools and workflows for developing and deploying models.
- **Education:** Documentation and tutorials that make technical concepts easier to understand.

## Technical toolkit

| Area | Technologies |
| --- | --- |
| Language & data | Python, Pandas, NumPy |
| Machine learning | PyTorch, TensorFlow, Scikit-learn |
| APIs & storage | FastAPI, PostgreSQL |
| Development & MLOps | Git, Docker, MLflow, Jupyter |

## Portfolio projects

These runnable portfolio demos include setup instructions, sample data, and automated tests. They demonstrate technical workflows; they are not deployed products or client work.

| Project | What you can explore | Stack |
| --- | --- | --- |
| [Grounded Document Assistant](https://github.com/kazmizaman865/kazmizaman865/tree/main/projects/document-assistant) | Ask questions through an API and receive relevant document passages with source citations. An extractive retrieval baseline for RAG. | Python, FastAPI, Scikit-learn |
| [Customer Churn Classifier](https://github.com/kazmizaman865/kazmizaman865/tree/main/projects/churn-classifier) | Train a reproducible model on synthetic data, inspect held-out metrics, and reload it for predictions. | Python, NumPy, Scikit-learn |
| [Feature Drift Monitor](https://github.com/kazmizaman865/kazmizaman865/tree/main/projects/drift-monitor) | Compare feature distributions, generate JSON reports, and trigger an automation alert on a simulated shift. | Python, NumPy, MLOps |

Start with a project's README for commands, design choices, and limitations. Each project runs locally without paid APIs or credentials.

## Try a project

```bash
git clone https://github.com/kazmizaman865/kazmizaman865.git
cd kazmizaman865/projects/document-assistant
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Use Python 3.12. On Windows, activate with `.venv\Scripts\activate` instead. Explore the running API at `/docs`, or follow the project README for example requests.

## Connect with me

For opportunities or collaboration related to AI engineering, machine learning, and data science, contact me through [LinkedIn](https://www.linkedin.com/in/sayed-zaman-ali-shah-917a1a222/) or [email](mailto:kazmizaman865@gmail.com).
