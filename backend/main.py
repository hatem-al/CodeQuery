"""
FastAPI application for RAG Code Documentation Assistant.
"""

import os
import re
import json
import logging
import urllib.parse
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
try:
    from pydantic import BaseModel, validator
except ImportError:
    from pydantic.v1 import BaseModel, validator
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from parser import parse_repo
from auth import (
    create_user,
    authenticate_user,
    create_access_token,
    decode_access_token,
    get_user_by_id,
    get_user_by_email,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from embeddings import (
    generate_embeddings,
    store_in_chromadb,
    search_code,
    hybrid_search_code,
    is_repo_indexed,
    get_chromadb_client,
    get_collection_for_repo,
    COLLECTION_NAME,
    load_indexed_repos,
    CHROMA_USING_INMEMORY,
)
from utils.query_processor import fix_common_typos, suggest_alternatives
from rag_engine import AdvancedRAG, QueryContext

_CONVERSATIONAL_PATTERNS = re.compile(
    r'^\s*('
    r'hi+|hey+|hello+|howdy|sup|yo|hiya|greetings|'
    r'how are you|how r u|what\'?s up|whats up|'
    r'good\s+(morning|afternoon|evening|day)|'
    r'thanks?(\s+you)?|thank\s+you|ty|thx|'
    r'ok+|okay|cool|great|nice|perfect|awesome|got\s+it|'
    r'what\s+can\s+you\s+do|what\s+do\s+you\s+do|help me'
    r')\s*[!?.]*\s*$',
    re.IGNORECASE
)

def _is_conversational(query: str) -> bool:
    return bool(_CONVERSATIONAL_PATTERNS.match(query.strip()))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter with IP-based key function
limiter = Limiter(key_func=get_remote_address)

# User-based rate limiting key function
def get_user_rate_limit_key(request: Request):
    """Get rate limit key based on user ID from token if authenticated, otherwise IP."""
    # Try to extract user from token in Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            payload = decode_access_token(token)
            if payload and "sub" in payload:
                return f"user:{payload['sub']}"
        except Exception:
            pass  # Fall back to IP if token is invalid
    return get_remote_address(request)

# Create user-based rate limiter
user_limiter = Limiter(key_func=get_user_rate_limit_key)

# Retry wrapper for OpenAI API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
async def call_openai_chat_with_retry(messages, model="gpt-4o", temperature=0.7, max_tokens=2000, stream=False):
    """
    Call OpenAI chat completion with retry logic.
    
    Args:
        messages: List of message dictionaries
        model: Model name (default: gpt-4o-mini)
        temperature: Temperature setting
        max_tokens: Maximum tokens
        stream: Whether to stream the response
    
    Returns:
        OpenAI API response
    """
    try:
        if stream:
            return await openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
        else:
            return await openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
    except Exception as e:
        logger.warning(f"OpenAI API call failed (will retry): {e}")
        raise

async def generate_hyde_document(query: str) -> str:
    """
    HyDE: generate a hypothetical code snippet that would answer the query,
    then use its embedding for retrieval (much closer to actual code vectors).
    Falls back to the original query on any error.
    """
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a short hypothetical code snippet (10-30 lines) "
                        "that would answer the following question. Output only code, no prose."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"HyDE generation failed, using original query: {e}")
        return query


# ---------------------------------------------------------------------------
# Shared RAG constants and helpers
# ---------------------------------------------------------------------------

_SCOPE_GUARD_PREFIX = "I can only answer questions about the indexed codebase"

_SYSTEM_PROMPT = """You are a code documentation assistant. Your job is to ANALYZE and EXPLAIN the implementation logic, business rules, and data flow in the code.

SCOPE RULE (check this first, before anything else):
If the user's question is NOT specifically about the code, implementation, or architecture of the indexed repository — for example, they are asking about project timelines, effort estimates, cost, team size, general advice, opinions, or anything else that cannot be answered by reading source code — respond ONLY with:
"I can only answer questions about the indexed codebase. Try asking how a feature is implemented, where something is defined, or how the code is structured."
Do NOT attempt to answer out-of-scope questions using the code snippets as a proxy.

CRITICAL INSTRUCTIONS:
1. EXPLAIN THE LOGIC — Don't just describe syntax. Explain WHY the code does what it does and HOW it implements business logic.
2. ANALYZE IMPLEMENTATION — Trace through the actual logic flow: what conditions are checked, what data is processed, what transformations occur.
3. SHOW THE CODE — Include actual code snippets in markdown blocks when explaining.
4. BE SPECIFIC — Instead of "handles errors", explain what the catch block actually does step by step.
5. IGNORE UI CODE — Skip React components, JSX, loading spinners, CSS. Focus on server logic, functions, data processing.
6. NO HEDGING — Avoid "likely", "probably", "seems". State what the code does based on what you see.
7. NO FABRICATION — If the snippets don't contain enough information, say "I don't see this implemented in the indexed code."
8. NO SPECULATION — Only describe what the indexed code actually does.
9. NO CONCLUSIONS — Do not add a "Conclusion" or "Summary" section.
10. CONVERSATION CONTEXT — Reference previous answers when relevant.

FORMATTING RULES:
- Use `backticks` for inline code — ALWAYS on the same line, never surrounded by newlines
- Multi-line code goes in ```language blocks``` ONLY
- Write in continuous paragraphs without random line breaks"""

_USER_PROMPT_TEMPLATE = """{clarification}Analyze the code snippets below and answer: {query}

{enhanced_context}

INSTRUCTIONS:
- Explain the implementation logic and data flow, not just syntax.
- Include code snippets in markdown blocks.
- Keep inline code on the same line: "The `User` class" not "The \\n\\nUser\\n class".
- Do NOT add a Conclusion or Summary. Do NOT speculate."""

# Token counting (exact with tiktoken, approximate fallback)
try:
    import tiktoken as _tiktoken
    _ENCODER = _tiktoken.encoding_for_model("gpt-4o")
    def _count_tokens(messages: list) -> int:
        return sum(len(_ENCODER.encode(m.get("content", ""))) + 4 for m in messages) + 2
except Exception:
    def _count_tokens(messages: list) -> int:
        return sum(len(m.get("content", "")) // 4 for m in messages)

_TOKEN_LIMIT = 90_000


async def _build_chat_payload(
    original_query: str,
    repo_id: str,
    chat_history: list,
    user_id: str,
) -> dict:
    """
    Shared RAG pipeline for /chat and /chat/stream.

    Returns a dict with:
      early_response (str | None) — set when results are empty; callers return this directly
      chat_messages  (list)       — ready-to-send messages for OpenAI
      sources_dict   (list[dict]) — for streaming SSE
      sources_obj    (list[Source]) — for non-streaming response
      depth          (int)        — query depth from QueryContext
    Raises HTTPException on configuration/repo errors.
    """
    corrected_query, was_corrected, corrections = fix_common_typos(original_query)
    query = corrected_query
    if was_corrected:
        logger.info(f"Typo correction: '{original_query}' → '{corrected_query}'")
    logger.info(f"Chat query: '{query[:100]}' (repo_id={repo_id})")

    # Repo verification
    if repo_id not in load_indexed_repos(user_id):
        raise HTTPException(status_code=404, detail=f"Repository not indexed: {repo_id}")

    try:
        collection = get_collection_for_repo(repo_id, user_id)
    except ValueError as e:
        msg = str(e)
        code = 404
        if any(k in msg.lower() for k in ("not found", "does not exist", "empty")):
            msg = f"Repository collection not found or empty. Please re-index: {repo_id}"
        raise HTTPException(status_code=code, detail=msg)

    # RAG: analyze → (conditional HyDE) → multi-hop search
    rag_engine = AdvancedRAG(hybrid_search_code, collection)
    query_context = rag_engine.analyze_query(query)
    logger.info(f"Query intent: {query_context.intent}, depth: {query_context.depth}")

    # HyDE only for deep queries — skip for fast location lookups (depth 1)
    if query_context.depth >= 2:
        query_context.hyde_document = await generate_hyde_document(query)

    search_results = rag_engine.multi_hop_search(query_context, top_k=8)

    if not search_results:
        return {
            "early_response": (
                "I couldn't find any relevant code for your question. "
                "Try rephrasing, or ask about a specific function, class, or feature."
            ),
            "chat_messages": [],
            "sources_dict": [],
            "sources_obj": [],
            "depth": query_context.depth,
        }

    low_confidence = [r for r in search_results if r.get("similarity", 0.0) < 0.3]

    # Organize and cap chunks by depth
    organized = rag_engine.organize_chunks(search_results)
    all_chunks: list = []
    for chunks in organized.get("by_file_type", {}).values():
        all_chunks.extend(chunks)
    for chunks in organized.get("by_content", {}).values():
        for c in chunks:
            if c not in all_chunks:
                all_chunks.append(c)
    if not all_chunks:
        all_chunks = search_results

    depth_cap = {1: 4, 2: 6, 3: 10}.get(query_context.depth, 6)
    all_chunks = [c for c in all_chunks if c.get("similarity", 0.0) >= 0.3][:depth_cap]
    if not all_chunks:
        all_chunks = search_results[:depth_cap]

    # Build sources
    sources_dict, sources_obj, context_parts = [], [], []
    for i, r in enumerate(all_chunks, 1):
        fp   = r["metadata"]["file"]
        ln   = r["metadata"]["lines"]
        code = r["code"]
        sim  = r.get("similarity", 0.0)
        lang = r["metadata"].get("language", "unknown")
        context_parts.append(f"--- Snippet {i} ({fp}, lines {ln}) ---\n{code}\n")
        sources_dict.append({"file": fp, "lines": ln, "code": code, "similarity": sim, "language": lang})
        sources_obj.append(Source(file=fp, lines=ln, code=code, similarity=sim, language=lang))

    # Clarification prefix
    clarification = ""
    if was_corrected:
        clarification = f"Corrected: {', '.join(f'{o}→{c}' for o, c in corrections)}. "
    if low_confidence and len(low_confidence) >= len(search_results) * 0.5:
        alts = suggest_alternatives(query, [r.get("metadata", {}).get("file", "") for r in low_confidence[:3]])
        if alts:
            clarification += f"Low-confidence results. Did you mean: {', '.join(alts[:3])}? "

    enhanced_context = rag_engine.build_prompt(query, organized, query_context)
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        clarification=clarification,
        query=query,
        enhanced_context=enhanced_context,
    )

    # Build messages with history
    messages: list = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for msg in (chat_history or [])[-6:]:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": user_prompt})

    # Token budget: drop trailing context chunks if over limit
    while _count_tokens(messages) > _TOKEN_LIMIT and len(context_parts) > 1:
        context_parts.pop()
        sources_dict.pop()
        sources_obj.pop()
        shorter = "\n".join(context_parts)
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            clarification=clarification,
            query=query,
            enhanced_context=shorter,
        )
        messages[-1] = {"role": "user", "content": user_prompt}

    logger.info(f"Prompt ~{_count_tokens(messages)} tokens, {len(all_chunks)} chunks, depth {query_context.depth}")

    return {
        "early_response": None,
        "chat_messages": messages,
        "sources_dict": sources_dict,
        "sources_obj": sources_obj,
        "depth": query_context.depth,
    }


# Initialize FastAPI app
app = FastAPI(
    title="RAG Code Documentation Assistant",
    description="API for indexing and querying code repositories",
    version="1.0.0"
)

# Add rate limiters to app
app.state.limiter = limiter
app.state.user_limiter = user_limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security
security = HTTPBearer()


# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

# Add CORS middleware for React frontend
# Get allowed origins from environment variable (comma-separated)
# Default to localhost for development
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client (async)
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory storage for indexing progress (in production, use Redis or database)
# TODO: For production, consider using Redis or SQLite for indexing progress
indexing_progress: Dict[str, Dict] = {}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring."""
    import embeddings as _emb
    storage_mode = "in-memory (data will be lost on restart)" if _emb.CHROMA_USING_INMEMORY else "persistent"
    return {
        "status": "degraded" if _emb.CHROMA_USING_INMEMORY else "healthy",
        "service": "RAG Code Documentation Assistant",
        "version": "1.0.0",
        "vector_storage": storage_mode,
    }


# Pydantic models for request/response
class RegisterRequest(BaseModel):
    email: str
    password: str
    username: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: Dict


class UserResponse(BaseModel):
    id: str
    email: str
    username: str


class IndexRequest(BaseModel):
    repo_url: str
    force_reindex: bool = False
    
    def __init__(self, **data):
        super().__init__(**data)
        # Validate repo_url length
        if len(self.repo_url) > 500:
            raise ValueError("repo_url must be 500 characters or less")


class IndexResponse(BaseModel):
    status: str
    chunks_indexed: int
    repo_id: str


class QueryRequest(BaseModel):
    query: str
    repo_id: str
    top_k: int = 5
    similarity_threshold: Optional[float] = 0.0
    language_filter: Optional[str] = None
    file_type_filter: Optional[str] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        # Validate input lengths
        if len(self.query) > 1000:
            raise ValueError("query must be 1000 characters or less")
        if len(self.repo_id) > 500:
            raise ValueError("repo_id must be 500 characters or less")
        # Validate top_k range
        if self.top_k < 1 or self.top_k > 50:
            raise ValueError("top_k must be between 1 and 50")
        # Validate similarity threshold
        if self.similarity_threshold < 0.0 or self.similarity_threshold > 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")


class QueryResult(BaseModel):
    file: str
    code: str
    metadata: Dict
    similarity: float


class QueryResponse(BaseModel):
    results: List[QueryResult]


class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    query: str
    repo_id: str
    chat_history: Optional[List[Dict[str, str]]] = []  # Previous conversation context
    
    class Config:
        # Allow extra fields to be ignored (for backwards compatibility)
        extra = "forbid"
    
    @validator('query')
    def validate_query(cls, v):
        if len(v) > 2000:
            raise ValueError("query must be 2000 characters or less")
        if not v or not v.strip():
            raise ValueError("query cannot be empty")
        return v
    
    @validator('repo_id')
    def validate_repo_id(cls, v):
        if len(v) > 500:
            raise ValueError("repo_id must be 500 characters or less")
        if not v or not v.strip():
            raise ValueError("repo_id cannot be empty")
        return v


class Source(BaseModel):
    file: str
    lines: str
    code: Optional[str] = None
    similarity: Optional[float] = None
    language: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]


class RepoInfo(BaseModel):
    repo_url: str
    indexed: bool


class ReposResponse(BaseModel):
    repos: List[RepoInfo]


@app.on_event("startup")
async def warmup():
    """Warmup on startup to reduce cold starts."""
    try:
        # Warm OpenAI connection
        logger.info("Application starting up...")
        logger.info("OpenAI client initialized")
        logger.info("ChromaDB client ready")
        logger.info("Application ready to accept requests")
    except Exception as e:
        logger.error(f"Error during startup warmup: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CodeQuery API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "POST /index": "Index a repository",
            "POST /query": "Query indexed code",
            "POST /chat": "Chat with codebase",
            "GET /repos": "List indexed repositories",
            "GET /health": "Health check",
            "GET /ping": "Quick ping check"
        }
    }


@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping():
    """Quick ping endpoint for health checks."""
    return {"status": "alive", "timestamp": __import__('datetime').datetime.now().isoformat()}


@app.post("/auth/register", response_model=TokenResponse)
@limiter.limit("5/minute")
async def register(request: Request, register_request: RegisterRequest):
    """Register a new user."""
    try:
        # Validate email format
        if '@' not in register_request.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format"
            )
        
        # Validate password strength
        password = register_request.password
        if len(password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )
        if len(password) > 128:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be less than 128 characters"
            )
        # Check for at least one letter and one number
        has_letter = any(c.isalpha() for c in password)
        has_number = any(c.isdigit() for c in password)
        if not (has_letter and has_number):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must contain at least one letter and one number"
            )
        
        # Create user
        user = create_user(
            email=register_request.email,
            password=register_request.password,
            username=register_request.username
        )
        
        # Create access token
        access_token = create_access_token(data={"sub": user["id"]})
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user={
                "id": user["id"],
                "email": user["email"],
                "username": user["username"]
            }
        )
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors, etc.)
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error registering user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/auth/login", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login(request: Request, login_request: LoginRequest):
    """Login and get access token."""
    user = authenticate_user(login_request.email, login_request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user["id"]})
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user={
            "id": user["id"],
            "email": user["email"],
            "username": user["username"]
        }
    )


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: Dict = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        username=current_user["username"]
    )


async def index_repository_background(repo_url: str, user_id: str, force_reindex: bool = False):
    """
    Background task to index a repository with progress tracking.
    """
    try:
        indexing_progress[repo_url] = {
            "status": "cloning",
            "stage": "Cloning repository...",
            "progress": 0,
            "chunks_indexed": 0,
            "error": None
        }
        
        # Check if already indexed
        if not force_reindex and is_repo_indexed(repo_url, user_id):
            indexing_progress[repo_url] = {
                "status": "already_indexed",
                "stage": "Already indexed",
                "progress": 100,
                "chunks_indexed": 0,
                "error": None
            }
            try:
                collection = get_collection_for_repo(repo_url, user_id)
                indexing_progress[repo_url]["chunks_indexed"] = collection.count()
            except Exception:
                pass
            return
        
        # Parse repository
        indexing_progress[repo_url].update({
            "status": "parsing",
            "stage": "Parsing code files...",
            "progress": 20
        })
        
        try:
            chunks, _ = parse_repo(repo_url, cleanup=True)
        except ValueError as ve:
            # Handle repository size/limit errors
            indexing_progress[repo_url].update({
                "status": "error",
                "error": str(ve)
            })
            return
        
        if not chunks:
            indexing_progress[repo_url].update({
                "status": "error",
                "error": "No code chunks found in repository"
            })
            return
        
        indexing_progress[repo_url].update({
            "status": "embedding",
            "stage": f"Generating embeddings for {len(chunks)} chunks...",
            "progress": 40,
            "chunks_indexed": len(chunks)
        })
        
        # Generate embeddings
        embeddings = generate_embeddings(chunks)
        
        indexing_progress[repo_url].update({
            "status": "storing",
            "stage": "Storing in ChromaDB...",
            "progress": 80
        })
        
        # Store in ChromaDB
        store_in_chromadb(chunks, embeddings, repo_url, user_id)
        
        indexing_progress[repo_url].update({
            "status": "completed",
            "stage": "Indexing completed!",
            "progress": 100,
            "chunks_indexed": len(chunks)
        })
        
    except Exception as e:
        logger.error(f"Error in background indexing: {e}", exc_info=True)
        indexing_progress[repo_url].update({
            "status": "error",
            "error": str(e)
        })


@app.post("/index", response_model=IndexResponse)
@limiter.limit("5/minute")  # Limit indexing to 5 requests per minute per IP
async def index_repository(
    request: Request, 
    index_request: IndexRequest, 
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """
    Index a GitHub repository by cloning, parsing, and embedding code chunks.
    Can run synchronously (default) or asynchronously with progress tracking.
    
    Args:
        request: IndexRequest with repo_url and optional force_reindex flag
        background_tasks: FastAPI background tasks
    
    Returns:
        IndexResponse with status, chunks_indexed, and repo_id
    """
    try:
        repo_url = index_request.repo_url
        force_reindex = index_request.force_reindex
        
        logger.info(f"Indexing repository: {repo_url} (force_reindex={force_reindex})")
        
        user_id = current_user["id"]
        
        # Check if already indexed
        if not force_reindex and is_repo_indexed(repo_url, user_id):
            logger.info(f"Repository already indexed: {repo_url} for user {user_id}")
            # Get collection to count chunks
            try:
                collection = get_collection_for_repo(repo_url, user_id)
                chunks_count = collection.count()
            except Exception:
                chunks_count = 0
            
            return IndexResponse(
                status="already_indexed",
                chunks_indexed=chunks_count,
                repo_id=repo_url
            )
        
        # Check if already indexing
        if repo_url in indexing_progress and indexing_progress[repo_url]["status"] in ["cloning", "parsing", "embedding", "storing"]:
            return IndexResponse(
                status="indexing",
                chunks_indexed=indexing_progress[repo_url].get("chunks_indexed", 0),
                repo_id=repo_url
            )
        
        # Run in background to avoid timeout
        background_tasks.add_task(index_repository_background, repo_url, user_id, force_reindex)
        
        # Initialize progress tracking
        indexing_progress[repo_url] = {
            "status": "starting",
            "stage": "Starting indexing...",
            "progress": 0,
            "chunks_indexed": 0,
            "error": None
        }
        
        logger.info(f"Started background indexing for {repo_url}")
        
        return IndexResponse(
            status="indexing",
            chunks_indexed=0,
            repo_id=repo_url
        )
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error indexing repository: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/index/status/{repo_url:path}")
async def get_indexing_status(repo_url: str, current_user: Dict = Depends(get_current_user)):
    """
    Get the indexing progress status for a repository.
    
    Args:
        repo_url: GitHub repository URL (URL-encoded)
    
    Returns:
        Progress status dictionary
    """
    repo_url = urllib.parse.unquote(repo_url)

    if repo_url in indexing_progress:
        return indexing_progress[repo_url]

    # Not in memory — server may have restarted. Check persistent indexed repos list.
    # If the repo is in the metadata file it completed successfully before the restart.
    try:
        from embeddings import load_indexed_repos, get_collection_for_repo, get_chromadb_client
        # We don't have a user_id here (unauthenticated endpoint), so scan all user files
        chroma_db_path = __import__('pathlib').Path(__file__).parent.parent / "data" / "chroma_db"
        chunks_count = 0
        found = False
        for meta_file in chroma_db_path.glob("indexed_repos_*.json"):
            user_id = meta_file.stem.replace("indexed_repos_", "")
            repos = load_indexed_repos(user_id)
            if repo_url in repos:
                found = True
                try:
                    collection = get_collection_for_repo(repo_url, user_id)
                    chunks_count = collection.count()
                except Exception:
                    pass
                break
        if found:
            return {
                "status": "completed",
                "stage": "Indexed (recovered after server restart)",
                "progress": 100,
                "chunks_indexed": chunks_count,
                "error": None
            }
    except Exception:
        pass

    raise HTTPException(
        status_code=404,
        detail="No indexing progress found for this repository"
    )


@app.post("/query", response_model=QueryResponse)
@user_limiter.limit("30/minute")  # Limit queries to 30 per minute per user
async def query_code(request: Request, query_request: QueryRequest, current_user: Dict = Depends(get_current_user)):
    """
    Search for relevant code chunks using semantic search.
    
    Args:
        request: QueryRequest with query, repo_id, and top_k
    
    Returns:
        QueryResponse with list of relevant code chunks
    """
    try:
        query = query_request.query
        repo_id = query_request.repo_id
        top_k = query_request.top_k
        
        logger.info(f"Querying: '{query}' (repo_id={repo_id}, top_k={top_k})")
        
        user_id = current_user["id"]
        
        # Verify repo is indexed
        if not is_repo_indexed(repo_id, user_id):
            raise HTTPException(
                status_code=404,
                detail=f"Repository not indexed: {repo_id}"
            )
        
        # Get repository-specific collection
        try:
            collection = get_collection_for_repo(repo_id, user_id)
        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Collection not found: {error_msg}")
            # Provide helpful error message
            if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                raise HTTPException(
                    status_code=404,
                    detail=f"Repository collection not found. This may happen if ChromaDB was restarted (in-memory mode). Please re-index the repository: {repo_id}"
                )
            raise HTTPException(
                status_code=404,
                detail=error_msg
            )
        
        # Search ChromaDB with filters
        results = search_code(
            query, 
            collection, 
            top_k=top_k,
            similarity_threshold=query_request.similarity_threshold or 0.0,
            language_filter=query_request.language_filter,
            file_type_filter=query_request.file_type_filter
        )
        
        # Format results
        query_results = []
        for result in results:
            query_results.append(QueryResult(
                file=result['metadata']['file'],
                code=result['code'],
                metadata=result['metadata'],
                similarity=result['similarity']
            ))
        
        logger.info(f"Found {len(query_results)} results")
        
        return QueryResponse(results=query_results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying code: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/chat/stream")
@user_limiter.limit("20/minute")  # Limit streaming chat to 20 per minute per user
async def chat_with_codebase_stream(
    request: Request, 
    chat_request: ChatRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Stream chat responses from the codebase using GPT-4o-mini.
    Uses Server-Sent Events (SSE) for real-time streaming.
    
    Args:
        request: ChatRequest with query and repo_id
    
    Returns:
        StreamingResponse with SSE events
    """
    async def generate_stream():
        try:
            original_query = chat_request.query
            repo_id = chat_request.repo_id

            if _is_conversational(original_query):
                yield f"data: {json.dumps({'sources': []})}\n\n"
                yield f"data: {json.dumps({'content': 'Hi! Ask me anything about the indexed codebase — how something works, where a function is defined, how a feature is implemented, etc.'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            try:
                payload = await _build_chat_payload(
                    original_query, repo_id,
                    chat_request.chat_history or [], current_user["id"]
                )
            except HTTPException as exc:
                yield f"data: {json.dumps({'error': exc.detail})}\n\n"
                return

            if payload["early_response"]:
                yield f"data: {json.dumps({'answer': payload['early_response'], 'sources': []})}\n\n"
                return

            yield f"data: {json.dumps({'sources': payload['sources_dict']})}\n\n"

            try:
                stream = await call_openai_chat_with_retry(
                    messages=payload["chat_messages"],
                    model="gpt-4o",
                    temperature=0.2,
                    max_tokens=3000,
                    stream=True,
                )
                accumulated = ""
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        accumulated += content
                        yield f"data: {json.dumps({'content': content})}\n\n"

                if accumulated.strip().startswith(_SCOPE_GUARD_PREFIX):
                    yield f"data: {json.dumps({'clear_sources': True})}\n\n"

                yield f"data: {json.dumps({'done': True})}\n\n"
                logger.info("Streaming completed")
            except Exception as e:
                logger.error(f"Error streaming from OpenAI: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': f'Error generating answer: {str(e)}'})}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming chat: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': f'Internal server error: {str(e)}'})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.post("/chat", response_model=ChatResponse)
@user_limiter.limit("20/minute")  # Limit chat to 20 per minute per user
async def chat_with_codebase(
    request: Request, 
    chat_request: ChatRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Chat with the codebase using GPT-4o-mini to generate answers based on retrieved code chunks.
    
    Args:
        request: ChatRequest with query and repo_id
    
    Returns:
        ChatResponse with answer and source citations
    """
    try:
        if not chat_request.query or not chat_request.repo_id:
            raise HTTPException(status_code=422, detail="Missing required fields: 'query' and 'repo_id'")

        original_query = chat_request.query
        repo_id = chat_request.repo_id

        if _is_conversational(original_query):
            return ChatResponse(
                answer="Hi! Ask me anything about the indexed codebase — how something works, where a function is defined, how a feature is implemented, etc.",
                sources=[]
            )

        payload = await _build_chat_payload(
            original_query, repo_id,
            chat_request.chat_history or [], current_user["id"]
        )

        if payload["early_response"]:
            return ChatResponse(answer=payload["early_response"], sources=[])

        try:
            response = await call_openai_chat_with_retry(
                messages=payload["chat_messages"],
                model="gpt-4o",
                temperature=0.2,
                max_tokens=3000,
            )
            answer = response.choices[0].message.content
            out_of_scope = answer.strip().startswith(_SCOPE_GUARD_PREFIX)
            return ChatResponse(
                answer=answer,
                sources=[] if out_of_scope else payload["sources_obj"],
            )
        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}", exc_info=True)
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded. Please wait and try again.")
            if "invalid_api_key" in err.lower() or "401" in err:
                raise HTTPException(status_code=500, detail="OpenAI API key is invalid.")
            if "insufficient_quota" in err.lower():
                raise HTTPException(status_code=500, detail="OpenAI quota exceeded.")
            raise HTTPException(status_code=500, detail=f"Error generating answer: {err}")

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/repos", response_model=ReposResponse)
async def list_repositories(current_user: Dict = Depends(get_current_user)):
    """
    List all indexed repositories for the current user.
    
    Returns:
        ReposResponse with list of indexed repositories
    """
    try:
        user_id = current_user["id"]
        indexed_repos = load_indexed_repos(user_id)
        
        repos = []
        for repo_url in indexed_repos:
            # Verify repo is actually indexed (collection exists)
            is_indexed = is_repo_indexed(repo_url, user_id)
            repos.append(RepoInfo(
                repo_url=repo_url,
                indexed=is_indexed
            ))
        
        logger.info(f"Found {len(repos)} indexed repositories")
        
        return ReposResponse(repos=repos)
        
    except Exception as e:
        logger.error(f"Error listing repositories: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.delete("/repos/{repo_url:path}")
async def delete_repository(repo_url: str, current_user: Dict = Depends(get_current_user)):
    """
    Delete a repository from the index.
    
    Args:
        repo_url: GitHub repository URL (URL-encoded)
    
    Returns:
        Success message
    """
    try:
        # Decode URL if needed
        repo_url = urllib.parse.unquote(repo_url)
        
        logger.info(f"Deleting repository: {repo_url}")
        
        user_id = current_user["id"]
        
        # Check if repo is in metadata (source of truth)
        indexed_repos = load_indexed_repos(user_id)
        if repo_url not in indexed_repos:
            raise HTTPException(
                status_code=404,
                detail=f"Repository not found in index: {repo_url}"
            )
        
        # Try to delete collection (may not exist in in-memory mode)
        try:
            client = get_chromadb_client()
            from embeddings import get_collection_name
            collection_name = get_collection_name(repo_url, user_id)
            try:
                # Check if collection exists before trying to delete
                client.get_collection(name=collection_name)
                client.delete_collection(name=collection_name)
                logger.info(f"Deleted collection: {collection_name}")
            except Exception as get_error:
                # Collection doesn't exist - that's OK, just log it
                logger.info(f"Collection {collection_name} doesn't exist (may be in-memory mode): {get_error}")
        except Exception as e:
            logger.warning(f"Error deleting collection: {e}")
            # Continue to remove from metadata even if collection deletion fails
        
        # Remove from indexed repos list
        indexed_repos = load_indexed_repos(user_id)
        if repo_url in indexed_repos:
            indexed_repos.remove(repo_url)
            from embeddings import get_indexed_repos_file
            metadata_file = get_indexed_repos_file(user_id)
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_file, 'w') as f:
                json.dump({'repos': indexed_repos}, f)
        
        logger.info(f"Successfully deleted repository: {repo_url}")
        
        return {
            "status": "deleted",
            "message": f"Repository {repo_url} has been deleted from the index"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting repository: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found in environment variables")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
