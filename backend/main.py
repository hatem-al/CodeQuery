"""
FastAPI application for RAG Code Documentation Assistant.
"""

import os
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
    is_repo_indexed,
    get_chromadb_client,
    get_collection_for_repo,
    COLLECTION_NAME,
    load_indexed_repos
)
from utils.query_processor import fix_common_typos, suggest_alternatives
from rag_engine import AdvancedRAG, QueryContext

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
async def call_openai_chat_with_retry(messages, model="gpt-4o-mini", temperature=0.7, max_tokens=2000, stream=False):
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
    return {
        "status": "healthy",
        "service": "RAG Code Documentation Assistant",
        "version": "1.0.0"
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


@app.get("/ping")
async def ping():
    """Quick ping endpoint for health checks."""
    return {"status": "alive", "timestamp": __import__('datetime').datetime.now().isoformat()}


@app.post("/auth/register", response_model=TokenResponse)
async def register(register_request: RegisterRequest):
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
async def login(login_request: LoginRequest):
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
async def get_indexing_status(repo_url: str):
    """
    Get the indexing progress status for a repository.
    
    Args:
        repo_url: GitHub repository URL (URL-encoded)
    
    Returns:
        Progress status dictionary
    """
    repo_url = urllib.parse.unquote(repo_url)
    
    if repo_url not in indexing_progress:
        raise HTTPException(
            status_code=404,
            detail="No indexing progress found for this repository"
        )
    
    return indexing_progress[repo_url]


@app.post("/query", response_model=QueryResponse)
@user_limiter.limit("30/minute")  # Limit queries to 30 per minute per user
async def query_code(request: Request, query_request: QueryRequest):
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
    import asyncio
    
    async def generate_stream():
        try:
            original_query = chat_request.query
            repo_id = chat_request.repo_id
            
            # Fix common typos in the query
            corrected_query, was_corrected, corrections = fix_common_typos(original_query)
            query = corrected_query  # Use corrected query for search
            
            if was_corrected:
                logger.info(f"Typo correction: '{original_query}' -> '{corrected_query}' (corrections: {corrections})")
            
            logger.info(f"Streaming chat query: '{query}' (repo_id={repo_id})")
            
            user_id = current_user["id"]
            
            # Verify repo is indexed
            if not is_repo_indexed(repo_id, user_id):
                yield f"data: {json.dumps({'error': f'Repository not indexed: {repo_id}'})}\n\n"
                return
            
            # Get repository-specific collection
            try:
                collection = get_collection_for_repo(repo_id, user_id)
            except ValueError as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return
            
            # Initialize Advanced RAG engine
            rag_engine = AdvancedRAG(search_code, collection)
            
            # Analyze query to determine intent and extract concepts
            query_context = rag_engine.analyze_query(query)
            logger.info(f"Streaming - Query intent: {query_context.intent}, main concept: {query_context.main_concept}, depth: {query_context.depth}")
            
            # Perform multi-hop search based on query context
            search_results = rag_engine.multi_hop_search(query_context, top_k=8)
            
            # Check for low confidence results
            low_confidence_results = []
            if search_results:
                for result in search_results:
                    similarity = result.get('similarity', 0.0)
                    if similarity < 0.3:
                        low_confidence_results.append(result)
            
            if not search_results:
                yield f"data: {json.dumps({'answer': 'I could not find any relevant code in the repository for your question. Try rephrasing your query or asking about a different topic.', 'sources': []})}\n\n"
                return
            
            # Organize chunks by file type and content type
            organized_chunks = rag_engine.organize_chunks(search_results)
            
            # Build context from organized chunks
            all_chunks = []
            for file_type_chunks in organized_chunks.get('by_file_type', {}).values():
                all_chunks.extend(file_type_chunks)
            for content_chunks in organized_chunks.get('by_content', {}).values():
                for chunk in content_chunks:
                    if chunk not in all_chunks:
                        all_chunks.append(chunk)
            
            # If organization didn't work well, fall back to original search results
            if not all_chunks:
                all_chunks = search_results
            
            # Limit to top results
            all_chunks = all_chunks[:12]
            
            context_parts = []
            sources = []
            
            for i, result in enumerate(all_chunks, 1):
                file_path = result['metadata']['file']
                lines = result['metadata']['lines']
                code = result['code']
                similarity = result.get('similarity', 0.0)
                language = result['metadata'].get('language', 'unknown')
                
                context_parts.append(
                    f"--- Code Snippet {i} (from {file_path}, lines {lines}) ---\n{code}\n"
                )
                sources.append({
                    'file': file_path,
                    'lines': lines,
                    'code': code,
                    'similarity': similarity,
                    'language': language
                })
            
            context = "\n".join(context_parts)
            
            # Build enhanced context using RAG engine
            enhanced_context = rag_engine.build_prompt(query, organized_chunks, query_context)
            
            # Build clarification message if needed
            clarification_message = ""
            if was_corrected:
                correction_terms = [f"'{orig}' → '{corr}'" for orig, corr in corrections]
                clarification_message = f"I think you meant '{corrected_query}' (corrected: {', '.join(correction_terms)}). "
            
            # Add suggestions for low confidence results
            suggestions = []
            if low_confidence_results and len(low_confidence_results) >= len(search_results) * 0.5:
                # More than 50% of results are low confidence
                suggestions = suggest_alternatives(query, [r.get('metadata', {}).get('file', '') for r in low_confidence_results[:3]])
                if suggestions:
                    clarification_message += f"The search results have low confidence. Did you mean: {', '.join(suggestions[:3])}? "
            
            # Build prompt
            system_prompt = """You are a code documentation assistant. Your job is to ANALYZE and EXPLAIN the implementation logic, business rules, and data flow in the code.

CRITICAL INSTRUCTIONS:
1. EXPLAIN THE LOGIC - Don't just describe syntax. Explain WHY the code does what it does and HOW it implements business logic.
2. ANALYZE IMPLEMENTATION - Trace through the actual logic flow: what conditions are checked, what data is processed, what transformations occur, what side effects happen.
3. UNDERSTAND BUSINESS RULES - Explain the actual rules, validations, error handling strategies, and algorithms implemented.
4. SHOW THE CODE - Include actual code snippets in markdown blocks when explaining.
5. BE SPECIFIC - Instead of "handles errors", explain "The catch block implements error handling by catching exceptions, logging them for debugging, and returning a structured JSON response with a 500 status code to inform the client of the failure."
6. FOCUS ON LOGIC - Explain the implementation details: how authentication works, how data is validated, how errors are handled, how business rules are enforced.
7. IGNORE UI CODE - Skip React components, JSX, loading spinners, CSS. Focus on server logic, functions, data processing, algorithms.
8. NO HEDGING - Avoid "likely", "probably", "seems", "appears", "we can infer". State what the code does based on what you see.
9. EXPLAIN PURPOSE - For each code block, explain what problem it solves and how it solves it.
10. TYPO ACKNOWLEDGMENT - If the query contained typos that were corrected, acknowledge the correction at the start of your response.
11. CLARIFICATION - If search results are unclear or have low confidence, ask the user for clarification or suggest alternative queries."""
            
            user_prompt = f"""{clarification_message}Analyze the code snippets below and answer: {query}

{enhanced_context}

INSTRUCTIONS:
1. EXPLAIN THE LOGIC - Analyze what the code implements, not just what syntax it uses. Explain the business logic, error handling strategy, data flow, and implementation approach.
2. SHOW CODE - Include code snippets in markdown blocks when explaining.
3. BE SPECIFIC - Explain HOW the code works: "The error handler catches exceptions during AI response parsing, logs the error for debugging, and returns a structured JSON response with status 500 to inform the client. The response includes both a user-friendly message and the raw AI response content for troubleshooting."
4. FOCUS ON IMPLEMENTATION - Explain the actual logic, algorithms, validations, and business rules, not just line-by-line syntax description.
5. IGNORE UI - Skip React/JSX/HTML/CSS. Focus on server logic, functions, data processing, algorithms.
6. EXPLAIN PURPOSE - For each code block, explain what problem it solves and how.
7. USE ORGANIZATION - The code is organized by file type and content type. Use this organization to provide a structured answer.

BAD: "Line 239 catches error. Line 240 returns response."
GOOD: "The error handler implements a strategy for handling AI parsing failures. When an exception occurs during AI response processing, it catches the error, constructs a JSON response with both a user-friendly error message and the raw AI response content (for debugging), and returns it with HTTP status 500 to indicate a server error."

Answer by analyzing the implementation logic and explaining how the code works."""
            
            # Send sources first
            yield f"data: {json.dumps({'sources': sources})}\n\n"
            
            # Stream response from GPT-4o-mini with retry logic
            logger.info("Streaming from GPT-4o-mini...")
            try:
                stream = await call_openai_chat_with_retry(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="gpt-4o-mini",
                temperature=0.2,  # Very low temperature for precise, literal reading
                max_tokens=3000,  # More tokens for detailed explanations
                    stream=True
                )
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield f"data: {json.dumps({'content': content})}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"
                logger.info("Streaming completed")
                
            except Exception as e:
                logger.error(f"Error streaming from OpenAI API: {e}", exc_info=True)
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
        # Validate request was parsed correctly
        if not chat_request.query or not chat_request.repo_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing required fields: 'query' and 'repo_id' are required"
            )
        
        original_query = chat_request.query
        repo_id = chat_request.repo_id
        
        # Fix common typos in the query
        corrected_query, was_corrected, corrections = fix_common_typos(original_query)
        query = corrected_query  # Use corrected query for search
        
        if was_corrected:
            logger.info(f"Typo correction: '{original_query}' -> '{corrected_query}' (corrections: {corrections})")
        
        logger.info(f"Chat query: '{query[:100]}...' (repo_id={repo_id})")
        
        user_id = current_user["id"]
        
        # Verify repo is indexed (check metadata file first)
        indexed_repos = load_indexed_repos(user_id)
        if repo_id not in indexed_repos:
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
            # Provide helpful error message for in-memory ChromaDB issue
            if "not found" in error_msg.lower() or "does not exist" in error_msg.lower() or "empty" in error_msg.lower():
                raise HTTPException(
                    status_code=404,
                    detail=f"Repository collection not found or empty. This may happen if ChromaDB was restarted (in-memory mode). Please re-index the repository: {repo_id}"
                )
            raise HTTPException(
                status_code=404,
                detail=error_msg
            )
        
        # Initialize Advanced RAG engine
        rag_engine = AdvancedRAG(search_code, collection)
        
        # Analyze query to determine intent and extract concepts
        query_context = rag_engine.analyze_query(query)
        logger.info(f"Query intent: {query_context.intent}, main concept: {query_context.main_concept}, depth: {query_context.depth}")
        
        # Perform multi-hop search based on query context
        search_results = rag_engine.multi_hop_search(query_context, top_k=8)
        
        # Check for low confidence results
        low_confidence_results = []
        if search_results:
            for result in search_results:
                similarity = result.get('similarity', 0.0)
                if similarity < 0.3:
                    low_confidence_results.append(result)
        
        if not search_results:
            logger.warning(f"No search results found for query: '{query}' in repo: {repo_id}")
            # Try again with no threshold to see if any results exist
            all_results = search_code(query, collection, top_k=5, similarity_threshold=0.0)
            if all_results:
                logger.info(f"Found {len(all_results)} results but all below 0.2 similarity threshold")
                return ChatResponse(
                    answer=f"I found some code, but it's not very relevant to your question (similarity scores were too low). Try rephrasing your query to be more specific. For example:\n- \"How does authentication work in this codebase?\"\n- \"Show me the main function\"\n- \"What classes are defined?\"\n\nYour query: \"{query}\"",
                    sources=[]
                )
            else:
                logger.warning(f"No results found even with threshold=0.0. Collection might be empty.")
                return ChatResponse(
                    answer="I couldn't find any relevant code in the repository for your question. This might mean:\n1. The repository hasn't been indexed yet (try indexing it first)\n2. Your query doesn't match any code in the repository\n3. The collection might be empty\n\nTry rephrasing your query or asking about a different topic. For example:\n- \"How does authentication work?\"\n- \"Show me the main function\"\n- \"What classes are defined in this codebase?\"",
                    sources=[]
                )
        
        # Organize chunks by file type and content type
        organized_chunks = rag_engine.organize_chunks(search_results)
        logger.info(f"Organized chunks: {len(organized_chunks.get('by_file_type', {}))} file types, {len(organized_chunks.get('by_content', {}))} content types")
        
        # Build context from organized chunks
        context_parts = []
        sources = []
        
        # Use organized chunks for better context
        all_chunks = []
        for file_type_chunks in organized_chunks.get('by_file_type', {}).values():
            all_chunks.extend(file_type_chunks)
        for content_chunks in organized_chunks.get('by_content', {}).values():
            for chunk in content_chunks:
                if chunk not in all_chunks:
                    all_chunks.append(chunk)
        
        # If organization didn't work well, fall back to original search results
        if not all_chunks:
            all_chunks = search_results
        
        # Limit to top results
        all_chunks = all_chunks[:12]  # Limit to 12 chunks max
        
        for i, result in enumerate(all_chunks, 1):
            file_path = result['metadata']['file']
            lines = result['metadata']['lines']
            code = result['code']
            similarity = result.get('similarity', 0.0)
            language = result['metadata'].get('language', 'unknown')
            
            context_parts.append(
                f"--- Code Snippet {i} (from {file_path}, lines {lines}) ---\n{code}\n"
            )
            sources.append(Source(
                file=file_path,
                lines=lines,
                code=code,
                similarity=similarity,
                language=language
            ))
        
        context = "\n".join(context_parts)
        
        # Build clarification message if needed
        clarification_message = ""
        if was_corrected:
            correction_terms = [f"'{orig}' → '{corr}'" for orig, corr in corrections]
            clarification_message = f"I think you meant '{corrected_query}' (corrected: {', '.join(correction_terms)}). "
        
        # Add suggestions for low confidence results
        suggestions = []
        if low_confidence_results and len(low_confidence_results) >= len(search_results) * 0.5:
            # More than 50% of results are low confidence
            suggestions = suggest_alternatives(query, [r.get('metadata', {}).get('file', '') for r in low_confidence_results[:3]])
            if suggestions:
                clarification_message += f"The search results have low confidence. Did you mean: {', '.join(suggestions[:3])}? "
        
        # Build conversation messages with history
        messages = [
            {"role": "system", "content": """You are a code documentation assistant. Your job is to ANALYZE and EXPLAIN the implementation logic, business rules, and data flow in the code.

CRITICAL INSTRUCTIONS:
1. EXPLAIN THE LOGIC - Don't just describe syntax. Explain WHY the code does what it does and HOW it implements business logic.
2. ANALYZE IMPLEMENTATION - Trace through the actual logic flow: what conditions are checked, what data is processed, what transformations occur, what side effects happen.
3. UNDERSTAND BUSINESS RULES - Explain the actual rules, validations, error handling strategies, and algorithms implemented.
4. SHOW THE CODE - Include actual code snippets in markdown blocks when explaining.
5. BE SPECIFIC - Instead of "handles errors", explain "The catch block implements error handling by catching exceptions, logging them for debugging, and returning a structured JSON response with a 500 status code to inform the client of the failure."
6. FOCUS ON LOGIC - Explain the implementation details: how authentication works, how data is validated, how errors are handled, how business rules are enforced.
7. IGNORE UI CODE - Skip React components, JSX, loading spinners, CSS. Focus on server logic, functions, data processing, algorithms.
8. NO HEDGING - Avoid "likely", "probably", "seems", "appears", "we can infer". State what the code does based on what you see.
9. EXPLAIN PURPOSE - For each code block, explain what problem it solves and how it solves it.
10. TYPO ACKNOWLEDGMENT - If the query contained typos that were corrected, acknowledge the correction at the start of your response.
11. CLARIFICATION - If search results are unclear or have low confidence, ask the user for clarification or suggest alternative queries.
12. CONVERSATION CONTEXT - Remember previous questions and answers in the conversation. Reference them when relevant (e.g., "As I mentioned earlier..." or "Building on the previous answer...")."""}
        ]
        
        # Add conversation history (limit to last 6 messages to avoid context overflow)
        chat_history = chat_request.chat_history or []
        if chat_history:
            # Take last 6 messages (3 exchanges)
            recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
            for msg in recent_history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add current query with code context
        user_prompt = f"""{clarification_message}Analyze the code snippets below and answer: {query}

CODE SNIPPETS:
{context}

INSTRUCTIONS:
1. EXPLAIN THE LOGIC - Analyze what the code implements, not just what syntax it uses. Explain the business logic, error handling strategy, data flow, and implementation approach.
2. SHOW CODE - Include code snippets in markdown blocks when explaining.
3. BE SPECIFIC - Explain HOW the code works: "The error handler catches exceptions during AI response parsing, logs the error for debugging, and returns a structured JSON response with status 500 to inform the client. The response includes both a user-friendly message and the raw AI response content for troubleshooting."
4. FOCUS ON IMPLEMENTATION - Explain the actual logic, algorithms, validations, and business rules, not just line-by-line syntax description.
5. IGNORE UI - Skip React/JSX/HTML/CSS. Focus on server logic, functions, data processing, algorithms.
6. EXPLAIN PURPOSE - For each code block, explain what problem it solves and how.
7. USE ORGANIZATION - The code is organized by file type and content type. Use this organization to provide a structured answer.
8. REFERENCE PREVIOUS CONTEXT - If this is a follow-up question, reference previous answers when relevant.

BAD: "Line 239 catches error. Line 240 returns response."
GOOD: "The error handler implements a strategy for handling AI parsing failures. When an exception occurs during AI response processing, it catches the error, constructs a JSON response with both a user-friendly error message and the raw AI response content (for debugging), and returns it with HTTP status 500 to indicate a server error."

Answer by analyzing the implementation logic and explaining how the code works."""
        
        messages.append({"role": "user", "content": user_prompt})
        
        # Call GPT-4o-mini with retry logic
        logger.info(f"Calling GPT-4o-mini with {len(messages)} messages (including {len(chat_history)} history messages)...")
        try:
            response = await call_openai_chat_with_retry(
                messages=messages,
                model="gpt-4o-mini",
                temperature=0.2,  # Very low temperature for precise, literal reading
                max_tokens=3000  # More tokens for detailed explanations
            )
            
            answer = response.choices[0].message.content
            
            logger.info("Successfully generated answer")
            
            return ChatResponse(
                answer=answer,
                sources=sources
            )
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
            error_msg = str(e)
            
            # Handle specific OpenAI errors
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                raise HTTPException(
                    status_code=429,
                    detail="OpenAI rate limit exceeded. Please wait a moment and try again."
                )
            elif "invalid_api_key" in error_msg.lower() or "401" in error_msg:
                raise HTTPException(
                    status_code=500,
                    detail="OpenAI API key is invalid. Please check your configuration."
                )
            elif "insufficient_quota" in error_msg.lower():
                raise HTTPException(
                    status_code=500,
                    detail="OpenAI API quota exceeded. Please check your account."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error generating answer: {error_msg}"
                )
        
    except HTTPException:
        raise
    except ValueError as e:
        # Handle validation errors from ChatRequest
        logger.error(f"Validation error in chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
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
