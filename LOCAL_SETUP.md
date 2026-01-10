# 🏠 Local Setup Guide

Complete guide for running CodeQuery locally on your machine.

## 📋 Prerequisites

Before you begin, make sure you have:

- ✅ **Python 3.9+** installed ([Download](https://www.python.org/downloads/))
- ✅ **Node.js 18+** installed ([Download](https://nodejs.org/))
- ✅ **Git** installed ([Download](https://git-scm.com/downloads))
- ✅ **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))

---

## 🚀 Step-by-Step Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/hatem-al/CodeQuery.git
cd CodeQuery
```

---

### 2️⃣ Backend Setup

#### **a) Navigate to backend directory**
```bash
cd backend
```

#### **b) Create and activate virtual environment**

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

#### **c) Install Python dependencies**
```bash
pip install -r requirements.txt
```

This will install:
- FastAPI, Uvicorn
- OpenAI, ChromaDB
- SQLAlchemy, bcrypt, PyJWT
- tree-sitter, GitPython
- And more...

#### **d) Create environment file**

**On macOS/Linux:**
```bash
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
JWT_SECRET_KEY=my-super-secret-jwt-key-change-this
ALLOWED_ORIGINS=http://localhost:5173
EOF
```

**On Windows (PowerShell):**
```powershell
@"
OPENAI_API_KEY=your_openai_api_key_here
JWT_SECRET_KEY=my-super-secret-jwt-key-change-this
ALLOWED_ORIGINS=http://localhost:5173
"@ | Out-File -FilePath .env -Encoding utf8
```

**Or manually create `backend/.env`:**
```env
OPENAI_API_KEY=sk-...your-key-here...
JWT_SECRET_KEY=my-super-secret-jwt-key-change-this
ALLOWED_ORIGINS=http://localhost:5173
```

⚠️ **Important:** Replace `your_openai_api_key_here` with your actual OpenAI API key!

#### **e) Run the backend**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

✅ **Backend is running at:** `http://localhost:8000`

Test it: Open `http://localhost:8000` in your browser - you should see:
```json
{"message": "RAG Code Documentation Assistant API", "status": "running"}
```

---

### 3️⃣ Frontend Setup

#### **a) Open a NEW terminal window**
Keep the backend running in the first terminal!

#### **b) Navigate to frontend directory**
```bash
cd frontend
```

#### **c) Install Node dependencies**
```bash
npm install
```

This will install:
- React, Vite
- Tailwind CSS
- Axios, Prism
- And more...

#### **d) Create environment file**

**On macOS/Linux:**
```bash
cat > .env << EOF
VITE_API_BASE_URL=http://localhost:8000
EOF
```

**On Windows (PowerShell):**
```powershell
@"
VITE_API_BASE_URL=http://localhost:8000
"@ | Out-File -FilePath .env -Encoding utf8
```

**Or manually create `frontend/.env`:**
```env
VITE_API_BASE_URL=http://localhost:8000
```

#### **e) Run the frontend**
```bash
npm run dev
```

You should see:
```
  VITE v7.3.0  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

✅ **Frontend is running at:** `http://localhost:5173`

---

## 🎉 You're Ready!

### **Open the app:**
1. Go to: `http://localhost:5173`
2. You should see the CodeQuery login page

### **Create an account:**
1. Click **"Don't have an account? Register"**
2. Enter email, password, and username
3. Click **Register**

### **Index your first repository:**
1. Enter a GitHub URL (e.g., `https://github.com/jinkscad/event-management-app`)
2. Click **"Index Repository"**
3. Wait 2-5 minutes for indexing to complete
4. The repo will appear in your indexed list

### **Ask questions:**
1. Select the indexed repository
2. Type a question like: "How does authentication work?"
3. Get AI-powered answers with code snippets!

---

## 🛑 Stopping the App

### **Stop Frontend:**
In the frontend terminal, press: `Ctrl + C`

### **Stop Backend:**
In the backend terminal, press: `Ctrl + C`

### **Deactivate Python virtual environment:**
```bash
deactivate
```

---

## 🔄 Restarting the App

### **Backend:**
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### **Frontend:**
```bash
cd frontend
npm run dev
```

---

## 📁 Data Storage

Your data is stored locally in:

- **User accounts:** `data/users.db` (SQLite database)
- **Indexed repos:** `data/chroma_db/` (ChromaDB vector database)
- **Chat history:** Browser localStorage (per repository)

---

## 🐛 Troubleshooting

### **"Command not found: python3"**
- Try `python` instead of `python3`
- Make sure Python is installed: `python --version`

### **"Command not found: npm"**
- Install Node.js from https://nodejs.org/

### **"Port 8000 already in use"**
- Another app is using port 8000
- Kill it: `lsof -ti:8000 | xargs kill -9` (macOS/Linux)
- Or use a different port: `uvicorn main:app --port 8001`

### **"Port 5173 already in use"**
- Another Vite app is running
- Kill it or Vite will suggest a different port

### **"OpenAI API key is invalid"**
- Check your `.env` file in the backend directory
- Make sure the key starts with `sk-`
- Get a new key at: https://platform.openai.com/api-keys

### **"Cannot connect to backend"**
- Make sure backend is running: `http://localhost:8000`
- Check `frontend/.env` has correct `VITE_API_BASE_URL`
- Restart both frontend and backend

---

## 💡 Tips

1. **Keep both terminals open** - one for backend, one for frontend
2. **Use `--reload` flag** - Backend auto-restarts on code changes
3. **Clear browser cache** - If you see old UI, hard refresh: `Ctrl+Shift+R`
4. **Check logs** - Backend terminal shows all API requests and errors
5. **Test the API** - Visit `http://localhost:8000/docs` for interactive API docs

---

## 🎥 Recording Your Demo

### **Recommended Tools:**

**macOS:**
- QuickTime Player (built-in)
- Screen Studio (paid, professional)
- OBS Studio (free)

**Windows:**
- Xbox Game Bar (built-in, `Win + G`)
- OBS Studio (free)
- Camtasia (paid)

**Linux:**
- SimpleScreenRecorder
- OBS Studio
- Kazam

### **What to Show:**

1. **Registration** - Create a new account
2. **Indexing** - Index a GitHub repository (show progress)
3. **Searching** - Ask 3-5 different questions
4. **Code Display** - Show syntax highlighting and source attribution
5. **Multiple Repos** - Index and switch between repositories
6. **Chat History** - Show conversation persistence

### **Tips:**

- Use a clean browser (no extensions visible)
- Zoom in on important parts
- Use a medium-sized repository (50-200 files)
- Prepare your questions beforehand
- Keep it under 5 minutes
- Add captions or voiceover explaining features

---

**Enjoy exploring codebases with AI! 🚀**

