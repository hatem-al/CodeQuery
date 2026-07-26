"""
Embeddings generation and vector storage using OpenAI and ChromaDB.
"""

import os
import re
import uuid
import json
import hashlib
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import chromadb
try:
    from chromadb.config import Settings
except ImportError:
    Settings = None


class _NoOpEmbeddingFunction:
    """Stops ChromaDB from loading SentenceTransformer — we always supply pre-computed vectors."""
    def __call__(self, texts):
        raise RuntimeError("Embeddings must be pre-computed; ChromaDB should not compute them.")

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ChromaDB persistent storage path
CHROMA_DB_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

# Base collection name prefix
COLLECTION_NAME_PREFIX = "code_docs"

def get_collection_name(repo_url: str, user_id: str) -> str:
    """
    Generate a unique collection name for a repository per user.
    Uses hash of repo URL and user ID to ensure uniqueness and valid collection name.
    
    Args:
        repo_url: GitHub repository URL
        user_id: User ID for isolation
    
    Returns:
        Collection name string
    """
    # Create a hash of the repo URL and user ID for uniqueness
    combined = f"{user_id}:{repo_url}"
    repo_hash = hashlib.md5(combined.encode()).hexdigest()[:12]
    # Clean repo URL for collection name (remove special chars, replace periods with underscores)
    repo_name = repo_url.replace('https://', '').replace('http://', '').replace('.git', '')
    # Replace periods and other special chars with underscores
    repo_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in repo_name)
    # Remove consecutive underscores
    while '__' in repo_name:
        repo_name = repo_name.replace('__', '_')
    # Remove leading/trailing underscores and hyphens
    repo_name = repo_name.strip('_-')
    # Limit length to ensure total name is under 63 chars
    repo_name = repo_name[:30]
    # Include user ID in collection name for isolation
    user_hash = hashlib.md5(user_id.encode()).hexdigest()[:8]
    collection_name = f"{COLLECTION_NAME_PREFIX}_{user_hash}_{repo_name}_{repo_hash}"
    # Ensure it starts and ends with alphanumeric
    if not collection_name[0].isalnum():
        collection_name = 'a' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'a'
    # Final length check (ChromaDB limit is 63 chars)
    if len(collection_name) > 63:
        collection_name = collection_name[:63]
        # Ensure it still ends with alphanumeric
        if not collection_name[-1].isalnum():
            collection_name = collection_name[:-1] + 'a'
    return collection_name

# Legacy collection name for backward compatibility
COLLECTION_NAME = "code_docs"

# Reuse a single client instance to support in-memory ChromaDB clients.
_CHROMA_CLIENT = None
# True when persistent init failed and we fell back to in-memory (data lost on restart)
CHROMA_USING_INMEMORY = False

# Cache query embeddings so repeated questions skip the OpenAI round-trip (~100ms saved each)
_QUERY_EMBEDDING_CACHE: Dict[str, List[float]] = {}
_QUERY_CACHE_MAX_SIZE = 500

# BM25 index cache: collection_name → (doc_count, bm25, ids, docs)
_BM25_CACHE: Dict[str, Tuple] = {}

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank_bm25 not installed. Hybrid search will use vector-only.")


def get_language_from_file(file_path: str) -> str:
    """
    Infer programming language from file extension.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Language name (e.g., 'python', 'javascript', 'typescript')
    """
    ext = Path(file_path).suffix.lower()
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
    }
    return language_map.get(ext, 'unknown')


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _get_embedding_batch(texts: List[str]) -> List[List[float]]:
    """
    Internal function to get embeddings for a batch of texts with retry logic.
    
    Args:
        texts: List of text strings to embed
    
    Returns:
        List of embedding vectors
    """
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise


def generate_embeddings(chunks: List[Dict[str, str]]) -> List[List[float]]:
    """
    Generate embeddings for code chunks using OpenAI's text-embedding-3-small model.
    
    Args:
        chunks: List of chunk dictionaries from parser.py
                Each dict has: code, file, type, name, lines
    
    Returns:
        List of embedding vectors matching the chunk order
    """
    if not chunks:
        return []
    
    embeddings = []
    batch_size = 100  # OpenAI allows up to 2048 inputs per request, but we'll batch at 100 for safety
    total_chunks = len(chunks)
    
    print(f"Generating embeddings for {total_chunks} chunks...")
    
    # Extract and clean code text from chunks
    texts = []
    for chunk in chunks:
        code = chunk.get('code', '')
        
        # Skip empty or whitespace-only code
        if not code or not code.strip():
            print(f"⚠ Warning: Skipping empty chunk from {chunk.get('file', 'unknown')}")
            code = "# Empty code chunk"  # Use placeholder instead of empty string
        
        # Truncate very long code (OpenAI limit is ~8191 tokens, roughly 32k chars)
        # We'll be conservative and limit to 20k characters
        if len(code) > 20000:
            print(f"⚠ Warning: Truncating long chunk from {chunk.get('file', 'unknown')} ({len(code)} chars)")
            code = code[:20000] + "\n# ... (truncated)"
        
        texts.append(code)
    
    # Process in batches
    for i in range(0, total_chunks, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} chunks)...")
        
        try:
            batch_embeddings = _get_embedding_batch(batch_texts)
            embeddings.extend(batch_embeddings)
            print(f"✓ Embedded {min(i + batch_size, total_chunks)}/{total_chunks} chunks...")
        except Exception as e:
            print(f"✗ Error in batch {batch_num}: {e}")
            # Fill with empty embeddings for failed batch (or re-raise)
            raise
    
    print(f"✓ Successfully generated {len(embeddings)} embeddings")
    return embeddings


def get_chromadb_client():
    """
    Get or create a ChromaDB client (PersistentClient for 0.4+, Client for 0.3.x).
    
    Returns:
        ChromaDB client instance
    
    Raises:
        RuntimeError: If ChromaDB cannot be initialized
    """
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is not None:
        return _CHROMA_CLIENT

    # Ensure the directory exists
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try PersistentClient (0.4+)
        if hasattr(chromadb, 'PersistentClient'):
            if Settings:
                _CHROMA_CLIENT = chromadb.PersistentClient(
                    path=str(CHROMA_DB_PATH),
                    settings=Settings(anonymized_telemetry=False)
                )
            else:
                _CHROMA_CLIENT = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            print(f"✓ Initialized persistent ChromaDB at {CHROMA_DB_PATH}")
        else:
            # ChromaDB 0.3.x API - use Client with Settings
            if Settings:
                _CHROMA_CLIENT = chromadb.Client(
                    Settings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=str(CHROMA_DB_PATH),
                        anonymized_telemetry=False
                    )
                )
                print(f"✓ Initialized persistent ChromaDB 0.3.x at {CHROMA_DB_PATH}")
            else:
                raise Exception("Settings module not available for ChromaDB 0.3.x")
    except Exception as e:
        # Last resort: in-memory client (WARNING: data will be lost on restart)
        global CHROMA_USING_INMEMORY
        import warnings
        warnings.warn(
            f"Failed to initialize persistent ChromaDB: {e}. "
            f"Falling back to in-memory mode. Data will be lost on server restart. "
            f"Please check ChromaDB installation and permissions for {CHROMA_DB_PATH}",
            RuntimeWarning
        )
        _CHROMA_CLIENT = chromadb.Client()
        CHROMA_USING_INMEMORY = True
        print(f"⚠ WARNING: Using in-memory ChromaDB. Data will not persist!")

    return _CHROMA_CLIENT


def get_indexed_repos_file(user_id: str) -> Path:
    """Get path to the indexed repos metadata file for a user."""
    return CHROMA_DB_PATH / f"indexed_repos_{user_id}.json"


def load_indexed_repos(user_id: str) -> List[str]:
    """Load list of indexed repository URLs from disk for a user."""
    metadata_file = get_indexed_repos_file(user_id)
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                return data.get('repos', [])
        except Exception:
            return []
    return []


def save_indexed_repo(repo_url: str, user_id: str):
    """Save repository URL to indexed repos list for a user."""
    metadata_file = get_indexed_repos_file(user_id)
    repos = load_indexed_repos(user_id)
    if repo_url not in repos:
        repos.append(repo_url)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump({'repos': repos}, f)


def remove_indexed_repo(repo_url: str, user_id: str):
    """Remove repository URL from indexed repos list for a user."""
    metadata_file = get_indexed_repos_file(user_id)
    repos = load_indexed_repos(user_id)
    if repo_url in repos:
        repos.remove(repo_url)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump({'repos': repos}, f)


def is_repo_indexed(repo_url: str, user_id: str) -> bool:
    """
    Check if a repository is already indexed in ChromaDB for a user.
    
    Args:
        repo_url: GitHub repository URL
        user_id: User ID
    
    Returns:
        True if repository is already indexed, False otherwise
    """
    try:
        # Check metadata file first
        indexed_repos = load_indexed_repos(user_id)
        if repo_url not in indexed_repos:
            return False
        
        # Try to verify collection exists (but don't fail if it doesn't)
        # This handles in-memory ChromaDB where collections might not persist
        try:
            client = get_chromadb_client()
            collection_name = get_collection_name(repo_url, user_id)
            collection = client.get_collection(name=collection_name, embedding_function=_NoOpEmbeddingFunction())
            # Check if collection has any documents
            count = collection.count()
            if count > 0:
                return True
            # Collection exists but is empty - still consider it indexed
            # (might be in the process of indexing)
            return True
        except Exception:
            # Collection doesn't exist - this is OK for in-memory ChromaDB
            # We trust the metadata file as the source of truth
            # In production with persistent ChromaDB, this would be an issue
            return True  # Trust metadata file
    except Exception:
        return False


def store_in_chromadb(chunks: List[Dict[str, str]], embeddings: List[List[float]], repo_url: str, user_id: str) -> Any:
    """
    Store code chunks and their embeddings in ChromaDB with persistent storage.
    Uses repository-specific collections for better isolation.
    
    Args:
        chunks: List of chunk dictionaries from parser.py
        embeddings: List of embedding vectors from generate_embeddings()
        repo_url: Repository URL to store in metadata
    
    Returns:
        ChromaDB Collection object
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
    
    print(f"Initializing ChromaDB at {CHROMA_DB_PATH}...")
    
    # Initialize persistent ChromaDB client
    client = get_chromadb_client()
    
    # Get repository-specific collection name (per user)
    collection_name = get_collection_name(repo_url, user_id)
    
    # Create or get collection
    try:
        # Try to get existing collection first
        try:
            collection = client.get_collection(name=collection_name, embedding_function=_NoOpEmbeddingFunction())
            print(f"✓ Using existing collection '{collection_name}'")
            # Clear existing collection if re-indexing
            print(f"  Clearing existing data for re-indexing...")
            client.delete_collection(name=collection_name)
            collection = client.create_collection(
                name=collection_name,
                embedding_function=_NoOpEmbeddingFunction(),
                metadata={
                    "description": f"Code documentation chunks for {repo_url}",
                    "repo_url": repo_url
                }
            )
            print(f"✓ Recreated collection '{collection_name}'")
        except Exception:
            collection = client.create_collection(
                name=collection_name,
                embedding_function=_NoOpEmbeddingFunction(),
                metadata={
                    "description": f"Code documentation chunks for {repo_url}",
                    "repo_url": repo_url
                }
            )
            print(f"✓ Created new collection '{collection_name}'")
        
        # Save repo URL to indexed repos list (per user)
        save_indexed_repo(repo_url, user_id)
    except Exception as e:
        print(f"Error accessing collection: {e}")
        raise
    
    # Prepare data for ChromaDB
    ids = []
    documents = []
    metadatas = []
    embedding_list = []
    
    print(f"Preparing {len(chunks)} chunks for storage...")
    
    for i, chunk in enumerate(chunks):
        # Generate unique ID: file:lines format
        chunk_id = f"{chunk['file']}:{chunk['lines']}"
        # If duplicate, append UUID to make it unique
        if chunk_id in ids:
            chunk_id = f"{chunk_id}:{str(uuid.uuid4())[:8]}"
        
        ids.append(chunk_id)
        documents.append(chunk['code'])
        embedding_list.append(embeddings[i])
        
        # Prepare metadata
        language = get_language_from_file(chunk['file'])
        metadatas.append({
            'file': chunk['file'],
            'type': chunk['type'],
            'name': chunk['name'],
            'lines': chunk['lines'],
            'language': language,
            'layer': chunk.get('layer', 'other'),
        })
    
    # Upsert to ChromaDB (handles duplicates)
    print(f"Storing chunks in ChromaDB...")
    try:
        collection.upsert(
            ids=ids,
            embeddings=embedding_list,
            documents=documents,
            metadatas=metadatas
        )
        print(f"✓ Successfully stored {len(chunks)} chunks in ChromaDB")
        
        # Persist data for ChromaDB 0.3.x (required for disk persistence)
        try:
            if hasattr(client, 'persist'):
                client.persist()
                print(f"✓ Persisted data to disk")
        except Exception as persist_error:
            print(f"⚠ Warning: Could not persist data: {persist_error}")
            
    except Exception as e:
        print(f"✗ Error storing in ChromaDB: {e}")
        raise
    
    return collection


def get_collection_for_repo(repo_url: str, user_id: str) -> Any:
    """
    Get the ChromaDB collection for a specific repository for a user.
    
    Args:
        repo_url: GitHub repository URL
        user_id: User ID
    
    Returns:
        ChromaDB Collection object
    
    Raises:
        ValueError: If repository is not indexed or collection doesn't exist
    """
    # Check metadata first
    indexed_repos = load_indexed_repos(user_id)
    if repo_url not in indexed_repos:
        raise ValueError(f"Repository not indexed: {repo_url}")
    
    client = get_chromadb_client()
    collection_name = get_collection_name(repo_url, user_id)
    
    try:
        collection = client.get_collection(name=collection_name, embedding_function=_NoOpEmbeddingFunction())
        # Verify collection has data
        if collection.count() == 0:
            raise ValueError(f"Collection exists but is empty for repository {repo_url}. Please re-index.")
        return collection
    except Exception as e:
        # Collection doesn't exist - this can happen with in-memory ChromaDB
        error_msg = str(e)
        if "does not exist" in error_msg.lower() or "not found" in error_msg.lower():
            raise ValueError(f"Collection not found for repository {repo_url}. The repository may need to be re-indexed. This can happen if ChromaDB was restarted (in-memory mode).")
        raise ValueError(f"Error accessing collection for repository {repo_url}: {e}")


def search_code(query: str, collection: Any, top_k: int = 5, similarity_threshold: float = 0.0, language_filter: Optional[str] = None, file_type_filter: Optional[str] = None, where_filter: Optional[Dict] = None) -> List[Dict]:
    """
    Search for relevant code chunks using semantic search with cosine similarity.
    
    Args:
        query: Natural language query string
        collection: ChromaDB Collection object
        top_k: Number of top results to return
    
    Returns:
        List of dictionaries with:
        - code: Code content
        - similarity: Similarity score (0-1, higher = better)
        - metadata: File, type, name, lines, language
        - id: Chunk ID
    """
    print(f"Searching for: '{query}'...")
    
    # Generate embedding for the query (use cache to avoid redundant API calls)
    try:
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in _QUERY_EMBEDDING_CACHE:
            query_embedding = _QUERY_EMBEDDING_CACHE[cache_key]
        else:
            query_embedding = _get_embedding_batch([query])[0]
            if len(_QUERY_EMBEDDING_CACHE) >= _QUERY_CACHE_MAX_SIZE:
                _QUERY_EMBEDDING_CACHE.pop(next(iter(_QUERY_EMBEDDING_CACHE)))
            _QUERY_EMBEDDING_CACHE[cache_key] = query_embedding
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        raise
    
    # Search ChromaDB with cosine similarity
    try:
        query_kwargs: Dict[str, Any] = {
            'query_embeddings': [query_embedding],
            'n_results': top_k,
            'include': ['documents', 'metadatas', 'distances']
        }
        if where_filter:
            query_kwargs['where'] = where_filter
        results = collection.query(**query_kwargs)
    except Exception as e:
        if where_filter:
            # where filter may fail if no docs match — retry without it
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances']
                )
            except Exception as e2:
                print(f"Error searching ChromaDB: {e2}")
                raise e2
        else:
            print(f"Error searching ChromaDB: {e}")
            raise
    
    # Format results with filtering
    retrieved_chunks = []
    
    if results['ids'] and len(results['ids'][0]) > 0:
        for i in range(len(results['ids'][0])):
            # ChromaDB returns distances (L2 or cosine)
            # For cosine distance: similarity = 1 - distance (already normalized)
            # For L2 distance: we normalize to [0, 1] range
            distance = results['distances'][0][i]
            
            # Convert distance to similarity score (0-1, higher = better)
            # ChromaDB 0.3.23 uses L2 by default, but we'll normalize it
            # Assuming distances are normalized, convert: similarity = 1 / (1 + distance)
            # Or if it's already cosine distance: similarity = 1 - distance
            # Try both approaches and use the one that gives reasonable scores
            if distance <= 1.0:
                # Likely cosine distance (0-1 range)
                similarity = max(0.0, min(1.0, 1 - distance))
            else:
                # Likely L2 distance, normalize using sigmoid-like function
                similarity = max(0.0, min(1.0, 1 / (1 + distance)))
            
            # Apply similarity threshold filter
            if similarity < similarity_threshold:
                continue
            
            metadata = results['metadatas'][0][i]
            
            # Apply language filter
            if language_filter and metadata.get('language', '').lower() != language_filter.lower():
                continue
            
            # Apply file type filter
            if file_type_filter and metadata.get('type', '').lower() != file_type_filter.lower():
                continue
            
            chunk_data = {
                'code': results['documents'][0][i],
                'metadata': metadata,
                'similarity': similarity,  # 0-1 scale, higher = better
                'id': results['ids'][0][i]
            }
            retrieved_chunks.append(chunk_data)
        
        print(f"✓ Found {len(retrieved_chunks)} relevant chunks (after filtering)")
    else:
        print("No results found")
    
    return retrieved_chunks


def _tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text for BM25: splits on whitespace/punctuation and camelCase."""
    tokens = re.findall(r'\w+', text.lower())
    expanded: List[str] = []
    for token in tokens:
        # split camelCase: fooBar → foo bar
        parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', token).lower().split()
        expanded.extend(parts)
    return expanded


def _get_bm25_index(collection: Any) -> Tuple[Any, List[str], List[str]]:
    """Return a cached BM25 index for the collection, rebuilding when docs change."""
    if not BM25_AVAILABLE:
        raise RuntimeError("rank_bm25 not installed")

    name = collection.name
    count = collection.count()
    cached = _BM25_CACHE.get(name)
    if cached and cached[0] == count:
        _, bm25, ids, docs = cached
        return bm25, ids, docs

    result = collection.get(include=['documents', 'ids'])
    ids: List[str] = result['ids']
    docs: List[str] = result['documents']
    tokenized = [_tokenize_for_bm25(d) for d in docs]
    bm25 = BM25Okapi(tokenized)
    _BM25_CACHE[name] = (count, bm25, ids, docs)
    return bm25, ids, docs


# Repo overview cache: collection_name → (doc_count, file listing)
_OVERVIEW_CACHE: Dict[str, Tuple[int, str]] = {}
_OVERVIEW_MAX_FILES = 200


def get_repo_overview(collection: Any) -> str:
    """
    Compact listing of every indexed file in the collection, for grounding
    broad questions ("what does this repo do?") that top-k retrieval alone
    can't answer. Cached per collection, rebuilt when the doc count changes.
    """
    try:
        name = getattr(collection, 'name', 'default')
        count = collection.count()
        cached = _OVERVIEW_CACHE.get(name)
        if cached and cached[0] == count:
            return cached[1]

        data = collection.get(include=['metadatas'])
        files = sorted({
            m.get('file') for m in (data.get('metadatas') or [])
            if isinstance(m, dict) and m.get('file')
        })
        if len(files) > _OVERVIEW_MAX_FILES:
            listing = '\n'.join(files[:_OVERVIEW_MAX_FILES])
            listing += f"\n... and {len(files) - _OVERVIEW_MAX_FILES} more files"
        else:
            listing = '\n'.join(files)

        _OVERVIEW_CACHE[name] = (count, listing)
        return listing
    except Exception as e:
        print(f"Failed to build repo overview: {e}")
        return ""


def _rrf_fuse(ranked_lists: List[List[str]], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion over multiple ranked ID lists."""
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def hybrid_search_code(
    query: str,
    collection: Any,
    top_k: int = 5,
    similarity_threshold: float = 0.0,
    language_filter: Optional[str] = None,
    file_type_filter: Optional[str] = None,
    where_filter: Optional[Dict] = None,
) -> List[Dict]:
    """
    Hybrid search: vector (cosine) + BM25 keyword, fused with Reciprocal Rank Fusion.
    Falls back to vector-only if rank_bm25 is unavailable.
    """
    fetch_k = top_k * 3

    # --- Vector search ---
    vector_results = search_code(
        query, collection,
        top_k=fetch_k,
        similarity_threshold=0.0,
        language_filter=language_filter,
        file_type_filter=file_type_filter,
        where_filter=where_filter,
    )
    vector_ids = [r['id'] for r in vector_results]

    if not BM25_AVAILABLE:
        # Apply threshold and return
        return [r for r in vector_results if r.get('similarity', 0) >= similarity_threshold][:top_k]

    # --- BM25 search ---
    try:
        bm25, all_ids, all_docs = _get_bm25_index(collection)
        query_tokens = _tokenize_for_bm25(query)
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(range(len(all_ids)), key=lambda i: scores[i], reverse=True)
        bm25_ids = [all_ids[i] for i in ranked[:fetch_k] if scores[i] > 0]
    except Exception as e:
        print(f"BM25 search failed, using vector-only: {e}")
        return [r for r in vector_results if r.get('similarity', 0) >= similarity_threshold][:top_k]

    # --- RRF fusion ---
    fused_ids = _rrf_fuse([vector_ids, bm25_ids])

    # Build lookup from already-fetched vector results
    result_by_id: Dict[str, Dict] = {r['id']: r for r in vector_results}

    # Fetch docs that appear only in BM25 results
    bm25_only = [doc_id for doc_id in fused_ids if doc_id not in result_by_id]
    if bm25_only:
        try:
            fetched = collection.get(ids=bm25_only, include=['documents', 'metadatas'])
            for i, doc_id in enumerate(fetched['ids']):
                meta = fetched['metadatas'][i]
                # Apply filters post-hoc
                if where_filter:
                    if not all(meta.get(k) == v for k, v in where_filter.items()):
                        continue
                if language_filter and meta.get('language', '').lower() != language_filter.lower():
                    continue
                if file_type_filter and meta.get('type', '').lower() != file_type_filter.lower():
                    continue
                result_by_id[doc_id] = {
                    'code': fetched['documents'][i],
                    'metadata': meta,
                    'similarity': 0.05,  # BM25-only baseline
                    'id': doc_id,
                    'retrieval': 'bm25',  # exempt from similarity thresholds downstream
                }
        except Exception as e:
            print(f"Failed to fetch BM25-only results: {e}")

    # Order by RRF rank, apply threshold
    ordered: List[Dict] = []
    for doc_id in fused_ids:
        if doc_id not in result_by_id:
            continue
        r = result_by_id[doc_id]
        if r.get('retrieval') != 'bm25' and r.get('similarity', 0) < similarity_threshold:
            continue
        ordered.append(r)
        if len(ordered) >= top_k:
            break

    print(f"✓ Hybrid search: {len(ordered)} results (vector={len(vector_ids)}, bm25={len(bm25_ids)})")
    return ordered


if __name__ == '__main__':
    import sys
    from parser import parse_repo
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in backend/.env file or export it.")
        sys.exit(1)
    
    # Parse command line arguments
    force_reindex = '--force' in sys.argv or '-f' in sys.argv
    if force_reindex:
        sys.argv = [arg for arg in sys.argv if arg not in ['--force', '-f']]
    
    # Test with smart_job_tracker repo or use provided URL
    if len(sys.argv) > 1:
        repo_url = sys.argv[1]
    else:
        repo_url = "https://github.com/hatem-al/smart_job_tracker.git"
    
    print("=" * 60)
    print("RAG Code Documentation Assistant - Embeddings Test")
    print("=" * 60)
    print(f"Repository: {repo_url}")
    print(f"Force reindex: {force_reindex}\n")
    
    # Check if repo is already indexed
    if not force_reindex and is_repo_indexed(repo_url):
        print(f"✓ Repository already indexed. Skipping embedding generation.")
        print(f"  Use --force or -f flag to re-index.\n")
        
        # Get collection for search (create if doesn't exist - needed for in-memory ChromaDB 0.3.23)
        client = get_chromadb_client()
        try:
            collection = client.get_collection(name=COLLECTION_NAME, embedding_function=_NoOpEmbeddingFunction())
            # Collection exists, proceed to search
        except Exception:
            # Collection doesn't exist (in-memory was cleared)
            print("  Note: Collection not found. Re-indexing required for search.")
            print("  Running with --force to re-index...\n")
            force_reindex = True
    
    if force_reindex or not is_repo_indexed(repo_url):
        # Step 1: Parse repository
        print("Step 1: Parsing repository...")
        try:
            chunks = parse_repo(repo_url)
            print(f"✓ Parsed {len(chunks)} code chunks\n")
        except Exception as e:
            print(f"✗ Error parsing repository: {e}")
            sys.exit(1)
        
        # Step 2: Generate embeddings
        print("Step 2: Generating embeddings...")
        try:
            embeddings = generate_embeddings(chunks)
            print()
        except Exception as e:
            print(f"✗ Error generating embeddings: {e}")
            sys.exit(1)
        
        # Step 3: Store in ChromaDB
        print("Step 3: Storing in ChromaDB...")
        try:
            collection = store_in_chromadb(chunks, embeddings, repo_url)
            print()
        except Exception as e:
            print(f"✗ Error storing in ChromaDB: {e}")
            sys.exit(1)
    
    # Step 4: Test search
    print("Step 4: Testing semantic search...")
    test_query = "How does authentication work?"
    print(f"Query: '{test_query}'\n")
    
    try:
        results = search_code(test_query, collection, top_k=5)
        print()
        
        if results:
            print("=" * 60)
            print("SEARCH RESULTS")
            print("=" * 60)
            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} (Similarity: {result['similarity']:.4f}) ---")
                print(f"File: {result['metadata']['file']}")
                print(f"Type: {result['metadata']['type']}")
                print(f"Name: {result['metadata']['name']}")
                print(f"Lines: {result['metadata']['lines']}")
                print(f"Language: {result['metadata']['language']}")
                print(f"\nCode preview:")
                code_preview = result['code'][:300] + "..." if len(result['code']) > 300 else result['code']
                print(code_preview)
        else:
            print("No results found.")
    except Exception as e:
        print(f"✗ Error during search: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ Test completed successfully!")
    print("=" * 60)
