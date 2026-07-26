"""
Advanced RAG Engine with multi-hop retrieval and query understanding.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict


@dataclass
class QueryContext:
    """Context information about a query."""
    intent: str  # 'architecture', 'location', 'usage'
    main_concept: str  # Primary concept being asked about
    related_concepts: List[str] = field(default_factory=list)  # Related concepts to search
    depth: int = 1  # How many hops to retrieve (1=direct, 2=one level, 3=deep)
    original_query: str = ""
    hyde_document: str = ""  # Hypothetical code snippet for HyDE retrieval


class AdvancedRAG:
    """Advanced RAG engine with multi-hop retrieval and query understanding."""
    
    def __init__(self, search_func, collection):
        """
        Initialize the RAG engine.
        
        Args:
            search_func: Function to search code (from embeddings.search_code)
            collection: ChromaDB collection to search
        """
        self.search_func = search_func
        self.collection = collection
    
    def analyze_query(self, query: str) -> QueryContext:
        """
        Analyze query to detect intent and extract concepts.
        
        Args:
            query: User's query string
            
        Returns:
            QueryContext with intent, main concept, related concepts, and depth
        """
        query_lower = query.lower()
        
        # Detect intent based on query patterns
        intent = 'architecture'  # default
        depth = 3  # default for architecture
        
        # Location queries: "where", "which file", "find", "locate"
        location_patterns = [
            r'\bwhere\b',
            r'\bwhich\s+file\b',
            r'\bfind\b',
            r'\blocate\b',
            r'\bpath\b',
            r'\blocation\b'
        ]
        if any(re.search(pattern, query_lower) for pattern in location_patterns):
            intent = 'location'
            depth = 1
        
        # Usage queries: "use", "usage", "example", "how to", "call"
        usage_patterns = [
            r'\buse\b',
            r'\busage\b',
            r'\bexample\b',
            r'\bhow\s+to\b',
            r'\bcall\b',
            r'\binvoke\b',
            r'\bimplement\b'
        ]
        if any(re.search(pattern, query_lower) for pattern in usage_patterns):
            intent = 'usage'
            depth = 2
        
        # Architecture queries: "how", "explain", "work", "flow", "structure"
        architecture_patterns = [
            r'\bhow\b',
            r'\bexplain\b',
            r'\bwork\b',
            r'\bflow\b',
            r'\bstructure\b',
            r'\barchitecture\b',
            r'\bdesign\b',
            r'\bmain\s+idea\b'
        ]
        if any(re.search(pattern, query_lower) for pattern in architecture_patterns):
            intent = 'architecture'
            depth = 3
        
        # Extract main concept (simplified - look for technical terms)
        main_concept = self._extract_main_concept(query)
        
        return QueryContext(
            intent=intent,
            main_concept=main_concept,
            related_concepts=[],
            depth=depth,
            original_query=query
        )
    
    def _extract_main_concept(self, query: str) -> str:
        """
        Extract the main concept from a query.
        
        Args:
            query: User's query string
            
        Returns:
            Main concept string
        """
        # Common technical terms to look for
        technical_terms = [
            'middleware', 'authentication', 'authorization', 'database', 'endpoint',
            'route', 'controller', 'service', 'model', 'component', 'function',
            'class', 'module', 'handler', 'validator', 'parser', 'config',
            'configuration', 'schema', 'migration', 'query', 'mutation',
            'resolver', 'api', 'rest', 'graphql', 'websocket', 'socket',
            'session', 'cookie', 'token', 'jwt', 'oauth', 'bcrypt', 'hash',
            'encryption', 'decryption', 'crypto', 'ssl', 'tls', 'https',
            'error', 'exception', 'logging', 'logger', 'debug', 'test',
            'spec', 'interface', 'type', 'enum', 'union', 'decorator',
            'middleware', 'guard', 'interceptor', 'filter', 'pipeline'
        ]
        
        query_lower = query.lower()
        
        # Find the most relevant technical term
        for term in technical_terms:
            if term in query_lower:
                return term
        
        # If no technical term found, try to extract noun phrases
        # Simple heuristic: take words after "the", "a", "an", or question words
        words = query.split()
        for i, word in enumerate(words):
            if word.lower() in ['the', 'a', 'an', 'what', 'which', 'where']:
                if i + 1 < len(words):
                    # Take next 1-2 words as concept
                    concept = ' '.join(words[i+1:i+3])
                    return concept.strip('?.,!')
        
        # Fallback: return first few words
        return ' '.join(words[:3]).strip('?.,!')
    
    def multi_hop_search(self, context: QueryContext, top_k: int = 8) -> List[Dict]:
        """
        Retrieve relevant chunks via hybrid search, preserving the fused
        relevance order returned by the search function.

        When a HyDE document is set it is used as the primary search query
        (better code-space proximity for embeddings); a second pass with the
        literal user query keeps keyword (BM25) grounding.
        """
        all_results = []
        seen_ids = set()

        def _add_results(results):
            for result in results:
                result_id = f"{result['metadata']['file']}:{result['metadata']['lines']}"
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    all_results.append(result)

        embed_query = context.hyde_document if context.hyde_document else context.original_query

        _add_results(self.search_func(
            embed_query,
            self.collection,
            top_k=top_k * 2,
            similarity_threshold=0.1
        ))

        # HyDE replaced the literal query above — run it too so exact
        # identifiers from the user's question still match via BM25
        if context.hyde_document:
            _add_results(self.search_func(
                context.original_query,
                self.collection,
                top_k=top_k,
                similarity_threshold=0.1
            ))

        return all_results[:top_k * 2]

    def organize_chunks(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Organize chunks by file type and content type.
        
        Args:
            chunks: List of code chunks
            
        Returns:
            Dictionary with organized chunks:
            {
                'by_file_type': { 'middleware': [...], 'routes': [...], ... },
                'by_content': { 'definitions': [...], 'usage': [...], 'config': [...] }
            }
        """
        organized = {
            'by_file_type': defaultdict(list),
            'by_content': defaultdict(list)
        }
        
        for chunk in chunks:
            file_path = chunk.get('metadata', {}).get('file', '')
            code = chunk.get('code', '').lower()
            
            # Organize by file type
            file_type = self._classify_file_type(file_path)
            organized['by_file_type'][file_type].append(chunk)
            
            # Organize by content type
            content_type = self._classify_content(code)
            organized['by_content'][content_type].append(chunk)
        
        return organized
    
    def _classify_file_type(self, file_path: str) -> str:
        """Classify file by its path."""
        path_lower = file_path.lower()
        
        if '/middleware' in path_lower or 'middleware' in path_lower:
            return 'middleware'
        elif '/route' in path_lower or '/api' in path_lower or '/endpoint' in path_lower:
            return 'routes'
        elif '/config' in path_lower or 'config' in path_lower:
            return 'config'
        elif '/model' in path_lower or '/schema' in path_lower:
            return 'models'
        elif '/service' in path_lower or '/util' in path_lower:
            return 'services'
        elif '/component' in path_lower or '/view' in path_lower:
            return 'components'
        elif file_path.endswith('index.js') or file_path.endswith('index.ts') or file_path.endswith('main.py'):
            return 'entry'
        else:
            return 'other'
    
    def _classify_content(self, code: str) -> str:
        """Classify code content type."""
        code_lower = code.lower()
        
        # Definition patterns
        if any(pattern in code_lower for pattern in [
            'function ', 'const ', 'class ', 'export ', 'def ', 'interface ',
            'type ', 'enum ', 'const ', 'let ', 'var '
        ]):
            if '=' in code_lower or ':' in code_lower:
                return 'definitions'
        
        # Usage patterns
        if any(pattern in code_lower for pattern in [
            '.use(', '.get(', '.post(', '.put(', '.delete(',
            'require(', 'import ', 'new ', '()'
        ]):
            return 'usage'
        
        # Configuration patterns
        if any(pattern in code_lower for pattern in [
            'config', 'setting', 'option', 'default', 'env',
            'process.env', 'app.use', 'middleware('
        ]):
            return 'config'
        
        return 'other'
