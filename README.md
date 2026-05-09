# 🧠 DocuMind — AI Document Q&A with CI/CD

> **Learn CI/CD, Docker, and RAG by building a real AI project!**

DocuMind is a RAG-powered (Retrieval-Augmented Generation) Document Q&A bot. Upload documents, ask questions, and get AI-generated answers grounded in YOUR data.

---

## 🎯 What You'll Learn

| Concept | How |
|---------|-----|
| **RAG Pipeline** | Chunk → Embed → Store → Retrieve → Generate |
| **Docker** | Containerize Python + AI dependencies |
| **CI/CD** | Auto-test on every push via GitHub Actions |
| **Testing AI Apps** | Mock LLM calls so tests run without API keys |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  DocuMind API                     │
│                                                   │
│  POST /documents/ingest                           │
│       │                                           │
│       ▼                                           │
│  ┌─────────┐   ┌──────────┐   ┌──────────────┐  │
│  │  Split   │──▶│  Embed   │──▶│  ChromaDB    │  │
│  │  Text    │   │ (local)  │   │ (vector DB)  │  │
│  └─────────┘   └──────────┘   └──────┬───────┘  │
│                                       │          │
│  POST /ask                            │          │
│       │                               │          │
│       ▼                               ▼          │
│  ┌─────────┐   ┌──────────┐   ┌──────────────┐  │
│  │  Embed  │──▶│ Retrieve │──▶│   Gemini     │  │
│  │ Question│   │ Top K    │   │   (LLM)      │  │
│  └─────────┘   └──────────┘   └──────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
cd cicd-learn
copy .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 2. Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Run Locally
```bash
flask --app app.main run --debug
```

### 4. Open the Chat UI
Visit: **http://localhost:5000/chat**

### 5. Or Use the API
```bash
# Health check
curl http://localhost:5000/

# Ingest a document
curl -X POST http://localhost:5000/documents/ingest ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"Python is a programming language...\", \"filename\": \"python.txt\"}"

# Ask a question
curl -X POST http://localhost:5000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What is Python?\"}"
```

---

## 🐳 Run with Docker

```bash
# Build the image
docker build -t documind:latest .

# Run the container
docker run -p 5000:5000 --env-file .env documind:latest

# Or use Docker Compose (recommended)
docker compose up --build
```

---

## 🧪 Run Tests

```bash
# Run all tests
pytest -v

# Run with coverage report
pytest --cov=app --cov-report=term-missing -v
```

---

## ⚙️ CI/CD Pipeline

The pipeline runs automatically on every push to `main`:

```
push → TEST → BUILD → DEPLOY
        │       │       │
     pytest   Docker   notification
     lint     health
              check
```

### How to trigger it:
1. Create a GitHub repository
2. Push this code to `main`
3. Go to **Actions** tab → Watch the pipeline run!

---

## 📁 Project Structure

```
cicd-learn/
├── app/                    # Application code
│   ├── main.py             # Flask API endpoints
│   ├── rag_engine.py       # RAG pipeline (the AI brain)
│   └── config.py           # Configuration management
├── tests/                  # Automated tests
│   ├── test_api.py         # API endpoint tests
│   ├── test_rag_engine.py  # RAG engine unit tests
│   └── test_config.py      # Configuration tests
├── static/
│   └── index.html          # Chat UI frontend
├── data/
│   └── sample.txt          # Sample document
├── .github/workflows/
│   └── ci-cd.yml           # GitHub Actions pipeline
├── Dockerfile              # Docker image recipe
├── docker-compose.yml      # Multi-service config
├── requirements.txt        # Python dependencies
└── .env.example            # Environment variables template
```

---

## 💰 Cost: $0

| Component | Technology | Cost |
|-----------|-----------|------|
| Embeddings | sentence-transformers | Free (local) |
| Vector DB | ChromaDB | Free (local) |
| LLM | Google Gemini Flash | Free tier |
| CI/CD | GitHub Actions | Free (2000 min/month) |
| Container | Docker | Free |
