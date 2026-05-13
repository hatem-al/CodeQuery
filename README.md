# CodeQuery

**Live at: https://codequery-frontend.onrender.com/**

An AI-powered code documentation assistant that lets you ask natural language questions about any GitHub repository. Paste a repo URL, wait for indexing, and start querying.

## How it works

1. Submit a GitHub URL — the backend clones the repo, parses all code files, generates embeddings via OpenAI, and stores them in ChromaDB
2. Ask a question — the backend classifies intent (location / usage / architecture), optionally generates a HyDE document for deep queries, runs a hybrid BM25 + vector search with RRF fusion, then streams a GPT-4o answer back in real time

## Stack

**Frontend:** React 19 + Vite + Tailwind CSS  
**Backend:** FastAPI + OpenAI API (embeddings + GPT-4o)  
**Storage:** SQLite (users) + ChromaDB (vectors)  
**Parsing:** Python AST + tree-sitter (JS/TS/Java/C++)

## Setup

### Prerequisites
- Python 3.9+
- Node.js 18+
- OpenAI API key

### Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create `backend/.env`:
```
OPENAI_API_KEY=sk-...
JWT_SECRET_KEY=your-secret-key-min-32-chars
ALLOWED_ORIGINS=http://localhost:5173
```

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env`:
```
VITE_API_BASE_URL=http://localhost:8000
```

```bash
npm run dev   # http://localhost:5173
```

### Tests

```bash
cd backend
python3 -m pytest   # 77 tests, no live services needed
```

## Features

- **Streaming responses** — real-time token streaming via SSE
- **Conversation history** — context from the last 6 messages is sent with each query
- **Multi-hop retrieval** — 1–3 search hops depending on query complexity
- **HyDE** — generates a hypothetical code snippet to improve vector search quality on architecture/usage queries
- **Hybrid search** — BM25 + cosine similarity fused with RRF
- **Per-user isolation** — separate ChromaDB collections and repo lists per account
- **Force re-index** — re-parse and re-embed any previously indexed repo
- **Dark mode** — persisted to localStorage

## Limits

| Resource | Limit |
|---|---|
| Repository size | 200 MB |
| Files per repo | 1 000 |
| Chunks per repo | 5 000 |
| Indexing requests | 5 / min per IP |
| Chat requests | 20 / min per user |

## API

| Method | Path | Description |
|---|---|---|
| POST | `/auth/register` | Create account |
| POST | `/auth/login` | Get JWT token |
| POST | `/index` | Trigger repo indexing |
| GET | `/index/status/{repo_url}` | Poll indexing progress |
| POST | `/chat/stream` | Streaming chat (SSE) |
| POST | `/chat` | Non-streaming chat |
| POST | `/query` | Raw semantic search |
| GET | `/repos` | List indexed repos |
| DELETE | `/repos/{repo_url}` | Remove a repo |

## Cost estimates (OpenAI)

- ~$0.10–$0.50 per repo indexed (scales with repo size)
- ~$0.01–$0.05 per chat query

## License

MIT
