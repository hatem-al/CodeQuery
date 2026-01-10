# CodeQuery - AI-Powered Code Search

CodeQuery is an intelligent code documentation assistant that uses RAG (Retrieval-Augmented Generation) to help you understand any GitHub repository through natural language queries.

![CodeQuery Demo](demo.gif) <!-- Add your video demo here -->

## ✨ Features

- 🔍 **Semantic Code Search** - Find code by meaning, not just keywords
- 🤖 **AI-Powered Explanations** - GPT-4 powered answers with code examples
- 📚 **Multi-Language Support** - Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more
- 🔐 **User Authentication** - Secure user accounts with JWT tokens
- 💾 **Persistent Storage** - SQLite database and ChromaDB vector storage
- ⚡ **Real-Time Streaming** - Server-sent events for instant chat responses
- 🎯 **Advanced RAG** - Multi-hop retrieval with query intent detection
- 📊 **Source Attribution** - See exactly where information comes from

## 🎥 Video Demo

[Watch the demo video](demo.mp4) <!-- Upload your video to the repo or YouTube -->

## 🏗️ Architecture

**Frontend:** React + Vite + Tailwind CSS + Josefin Sans  
**Backend:** FastAPI (Python) + OpenAI API  
**Database:** SQLite (users) + ChromaDB (embeddings)  
**Parsing:** Python AST + tree-sitter for multi-language support

## 📋 Prerequisites

- Python 3.9+
- Node.js 18+
- OpenAI API Key ([Get one here](https://platform.openai.com/api-keys))
- Git

## 🚀 Quick Start (Local Setup)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/hatem-al/CodeQuery.git
cd CodeQuery
```

### 2️⃣ Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
JWT_SECRET_KEY=your_secret_key_here_change_in_production
ALLOWED_ORIGINS=http://localhost:5173
EOF

# Run the backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at: `http://localhost:8000`

### 3️⃣ Frontend Setup

Open a **new terminal** window:

```bash
cd frontend

# Install dependencies
npm install

# Create .env file
cat > .env << EOF
VITE_API_BASE_URL=http://localhost:8000
EOF

# Run the frontend
npm run dev
```

The frontend will be available at: `http://localhost:5173`

### 4️⃣ Start Using CodeQuery

1. Open `http://localhost:5173` in your browser
2. **Register** a new account
3. **Index a repository** (e.g., `https://github.com/username/repo`)
4. Wait for indexing to complete (~2-5 minutes)
5. **Ask questions** about the code!

## 💡 Example Queries

- "How does authentication work in this codebase?"
- "Show me the main API endpoints"
- "Explain the database schema"
- "What error handling strategies are used?"
- "How is data validation implemented?"
- "Show me all the middleware functions"

## 🛠️ Tech Stack

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Axios** - HTTP client
- **Josefin Sans** - Custom font

### Backend
- **FastAPI** - Modern Python web framework
- **OpenAI API** - `text-embedding-3-small` for embeddings, `gpt-4o-mini` for chat
- **ChromaDB** - Vector database for semantic search
- **SQLAlchemy** - ORM for SQLite
- **tree-sitter** - Multi-language code parser
- **GitPython** - GitHub repository cloning
- **bcrypt** - Password hashing
- **PyJWT** - JWT token authentication
- **slowapi** - Rate limiting

## 📁 Project Structure

```
CodeQuery/
├── backend/
│   ├── main.py              # FastAPI app and API endpoints
│   ├── auth.py              # Authentication logic
│   ├── database.py          # SQLite database setup
│   ├── embeddings.py        # OpenAI embeddings & ChromaDB
│   ├── parser.py            # Code parsing (AST + tree-sitter)
│   ├── retrieval.py         # Semantic search logic
│   ├── rag_engine.py        # Advanced RAG with multi-hop search
│   ├── utils/
│   │   └── query_processor.py  # Typo detection & query processing
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main app component
│   │   ├── components/      # React components
│   │   │   ├── Header.jsx
│   │   │   ├── Login.jsx
│   │   │   ├── EmptyState.jsx
│   │   │   ├── RepoInput.jsx
│   │   │   ├── ChatInterface.jsx
│   │   │   ├── CodeBlock.jsx
│   │   │   ├── SourcesList.jsx
│   │   │   └── LoadingSpinner.jsx
│   │   └── main.jsx
│   ├── package.json         # Node dependencies
│   └── vite.config.js       # Vite configuration
├── data/
│   ├── chroma_db/           # ChromaDB vector storage (auto-created)
│   └── users.db             # SQLite database (auto-created)
└── README.md
```

## ⚙️ Configuration

### Backend Environment Variables

Create `backend/.env`:

```env
# Required
OPENAI_API_KEY=sk-...                          # Your OpenAI API key

# Optional (defaults shown)
JWT_SECRET_KEY=your-secret-key-here            # JWT signing key
ALLOWED_ORIGINS=http://localhost:5173          # CORS origins (comma-separated)
```

### Frontend Environment Variables

Create `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

## 🎨 Features Deep Dive

### 1. Advanced RAG Engine
- **Query Intent Detection** - Classifies queries as architecture, location, or usage questions
- **Multi-Hop Search** - Iteratively searches for related concepts
- **Concept Extraction** - Identifies related terms from retrieved code
- **Smart Organization** - Groups code by file type and content type

### 2. Typo Detection
- Automatically corrects common programming term misspellings
- Suggests alternatives for low-confidence search results

### 3. Repository Size Limits
- **Max Repository Size:** 200 MB
- **Max Files:** 1000
- **Max Code Chunks:** 5000
- Prevents timeouts and excessive API costs

### 4. User Isolation
- Each user has separate indexed repositories
- ChromaDB collections are user-specific
- JWT-based authentication

## 🔒 Security

- ✅ Password hashing with bcrypt
- ✅ JWT token authentication
- ✅ Rate limiting (slowapi)
- ✅ CORS configuration
- ✅ Input validation (Pydantic)
- ✅ SQL injection protection (SQLAlchemy ORM)

## 📊 Performance

- **Indexing Speed:** ~2-5 minutes for medium repos (50-200 files)
- **Search Latency:** <500ms for semantic search
- **Chat Response:** Streaming (real-time tokens)
- **Concurrent Users:** Supports multiple users with rate limiting

## 🐛 Troubleshooting

### "OpenAI rate limit exceeded"
- You've hit OpenAI's API rate limit
- Wait a few minutes and try again
- Consider upgrading your OpenAI plan

### "Repository too large"
- The repository exceeds size limits (200 MB, 1000 files, or 5000 chunks)
- Try a smaller repository
- Adjust limits in `backend/parser.py` and `backend/main.py`

### "ChromaDB collection not found"
- The repository needs to be re-indexed
- Click "Force Re-index" on the repository

### "Cannot connect to backend"
- Make sure the backend is running: `http://localhost:8000`
- Check that the `VITE_API_BASE_URL` in `frontend/.env` is correct

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Hatem Almasri**

## 🙏 Acknowledgments

- OpenAI for GPT-4 and embeddings API
- ChromaDB for vector storage
- FastAPI for the excellent Python web framework
- React and Vite for the frontend tooling

---

## 📝 Notes

- The first indexing of a repository will download and parse all code files
- Embeddings are cached in ChromaDB for fast subsequent queries
- Chat history is stored in browser localStorage (per repository)
- The app uses OpenAI's API, so you'll incur costs based on usage
  - ~$0.10-0.50 per repository indexing (depending on size)
  - ~$0.01-0.05 per chat query

---

**Enjoy exploring codebases with AI! 🚀**
