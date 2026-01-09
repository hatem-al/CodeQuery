# CodeQuery

> AI-powered semantic search for codebases using Retrieval-Augmented Generation

CodeQuery enables natural language search and exploration of GitHub repositories. The system indexes codebases, generates vector embeddings, and leverages GPT-4 to provide intelligent answers with relevant code context.

## Features

- **Intelligent Indexing**: Automatically clone and index public GitHub repositories
- **Natural Language Query**: Ask questions about code in plain English
- **Contextual Responses**: Receive answers with relevant code snippets, file paths, and line numbers
- **Multi-User Architecture**: Isolated user accounts with separate repository collections
- **Multi-Language Support**: Python, JavaScript, TypeScript, Java, C++, and more
- **Persistent Chat History**: Maintain conversation context across sessions
- **Advanced RAG Pipeline**: Multi-hop retrieval with typo correction and query clarification

## Architecture

### Frontend
- React with Vite
- Tailwind CSS for styling
- Syntax highlighting via Prism

### Backend
- FastAPI (Python 3.11)
- ChromaDB for vector storage
- OpenAI API for embeddings and completions
- SQLite for user management
- Tree-sitter for multi-language code parsing

## Installation

### Prerequisites

- Node.js 18 or higher
- Python 3.11 or higher
- OpenAI API key

### Local Development Setup

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/codequery.git
cd codequery
```

**2. Configure the backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Set environment variables**

Create a `.env` file in the `backend` directory:
```env
OPENAI_API_KEY=your-openai-api-key
SECRET_KEY=your-jwt-secret-key
```

**4. Start the backend server**
```bash
uvicorn main:app --reload --port 8000
```

**5. Configure the frontend** (in a new terminal)
```bash
cd frontend
npm install
```

Create a `.env` file in the `frontend` directory:
```env
VITE_API_BASE_URL=http://localhost:8000
```

**6. Start the frontend development server**
```bash
npm run dev
```

**7. Access the application**

Navigate to `http://localhost:5173` in your browser.

## Deployment

### Vercel (Frontend)

1. Connect your GitHub repository to Vercel
2. Configure the following settings:
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `dist`
3. Set environment variable: `VITE_API_BASE_URL` (your backend URL)

### Render (Backend)

1. Create a new Web Service on Render
2. Configure the following settings:
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Set environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `SECRET_KEY`: Random string for JWT token signing
4. Add persistent disk (1GB minimum) mounted at `/opt/render/project/src/data`

## Usage

1. Register a new account or log in
2. Enter a GitHub repository URL (e.g., `https://github.com/facebook/react`)
3. Wait for the indexing process to complete
4. Query the codebase using natural language

**Example Queries:**
- "How does authentication work in this application?"
- "Where is the database connection initialized?"
- "Explain the main application entry point"
- "Show me examples of API endpoint definitions"

## Configuration

### Supported Programming Languages

- Python (`.py`)
- JavaScript (`.js`, `.jsx`)
- TypeScript (`.ts`, `.tsx`)
- Java (`.java`)
- C/C++ (`.c`, `.cpp`, `.h`, `.hpp`)

### AI Models

- **Embeddings**: `text-embedding-3-small` (OpenAI)
- **Chat Completions**: `gpt-4o-mini` (OpenAI)

## Technical Details

### RAG Pipeline

The system implements an advanced Retrieval-Augmented Generation pipeline:

1. **Code Parsing**: Multi-language parsing using Tree-sitter
2. **Chunking**: Intelligent code segmentation preserving semantic boundaries
3. **Embedding**: Vector generation using OpenAI's embedding model
4. **Storage**: Vector storage and similarity search via ChromaDB
5. **Retrieval**: Multi-hop retrieval with query expansion
6. **Generation**: Context-aware response generation using GPT-4

### Security

- JWT-based authentication
- Bcrypt password hashing
- User-isolated data storage
- API rate limiting
- Environment-based configuration

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Submit a pull request with a detailed description

For bug reports or feature requests, please open an issue on GitHub.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

Created by Hatem Almasri

## Acknowledgments

- OpenAI for GPT-4 and embedding models
- ChromaDB for vector database capabilities
- Tree-sitter for language-agnostic code parsing
- The FastAPI and React communities

---
