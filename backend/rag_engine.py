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
        Perform multi-hop search based on query context.
        
        Args:
            context: QueryContext with intent and concepts
            top_k: Number of results per hop
            
        Returns:
            List of code chunks from multi-hop search
        """
        all_results = []
        seen_ids = set()
        
        # First hop: search for main concept
        main_results = self.search_func(
            context.main_concept,
            self.collection,
            top_k=top_k,
            similarity_threshold=0.2
        )
        
        for result in main_results:
            result_id = f"{result['metadata']['file']}:{result['metadata']['lines']}"
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                all_results.append(result)
        
        # Extract related concepts from first hop results
        if context.depth > 1 and main_results:
            related_concepts = self._extract_related_concepts(main_results)
            context.related_concepts = related_concepts[:5]  # Limit to 5 related concepts
        
        # Second hop: search for related concepts
        if context.depth >= 2 and context.related_concepts:
            for concept in context.related_concepts:
                concept_results = self.search_func(
                    concept,
                    self.collection,
                    top_k=top_k // 2,  # Fewer results per related concept
                    similarity_threshold=0.2
                )
                
                for result in concept_results:
                    result_id = f"{result['metadata']['file']}:{result['metadata']['lines']}"
                    if result_id not in seen_ids:
                        seen_ids.add(result_id)
                        all_results.append(result)
        
        # Third hop: search for usage/configuration patterns
        if context.depth >= 3:
            # Search for patterns like "how X is used", "X configuration"
            usage_queries = [
                f"{context.main_concept} usage",
                f"{context.main_concept} configuration",
                f"{context.main_concept} setup",
                f"how {context.main_concept} works"
            ]
            
            for usage_query in usage_queries[:2]:  # Limit to 2 usage queries
                usage_results = self.search_func(
                    usage_query,
                    self.collection,
                    top_k=top_k // 4,
                    similarity_threshold=0.2
                )
                
                for result in usage_results:
                    result_id = f"{result['metadata']['file']}:{result['metadata']['lines']}"
                    if result_id not in seen_ids:
                        seen_ids.add(result_id)
                        all_results.append(result)
        
        # Sort by similarity (highest first)
        all_results.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
        
        # Limit total results
        return all_results[:top_k * 2]  # Allow more results for multi-hop
    
    def _extract_related_concepts(self, chunks: List[Dict]) -> List[str]:
        """
        Extract related concepts from code chunks.
        
        Args:
            chunks: List of code chunks from search results
            
        Returns:
            List of related concept strings
        """
        concepts = set()
        
        for chunk in chunks:
            code = chunk.get('code', '')
            metadata = chunk.get('metadata', {})
            file_path = metadata.get('file', '')
            
            # Extract imports/requires
            import_patterns = [
                r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',  # ES6 imports
                r'require\([\'"]([^\'"]+)[\'"]\)',  # CommonJS requires
                r'from\s+[\'"]([^\'"]+)[\'"]',  # Python imports
                r'import\s+([^\s]+)',  # Simple imports
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, code, re.IGNORECASE)
                for match in matches:
                    # Extract module name (last part of path)
                    module = match.split('/')[-1].split('.')[0]
                    if module and len(module) > 2:
                        concepts.add(module)
            
            # Extract function calls (simplified)
            function_call_pattern = r'(\w+)\s*\([^)]*\)'
            function_calls = re.findall(function_call_pattern, code)
            for func in function_calls[:10]:  # Limit to first 10
                if func and len(func) > 2 and not func[0].islower() or func in ['use', 'get', 'set', 'has', 'is']:
                    # Skip common words, keep technical terms
                    if func not in ['if', 'for', 'while', 'return', 'const', 'let', 'var']:
                        concepts.add(func)
            
            # Extract class names
            class_pattern = r'class\s+(\w+)'
            classes = re.findall(class_pattern, code)
            for cls in classes:
                concepts.add(cls)
            
            # Extract from file path (directory names often indicate concepts)
            path_parts = file_path.split('/')
            for part in path_parts:
                if part and part not in ['src', 'lib', 'app', 'components', 'utils', 'helpers']:
                    if len(part) > 2:
                        concepts.add(part)
        
        # Filter and return most relevant concepts
        filtered_concepts = [
            c for c in concepts 
            if len(c) > 2 and c.lower() not in ['the', 'and', 'or', 'not', 'for', 'with']
        ]
        
        return list(filtered_concepts)[:10]  # Return top 10
    
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
    
    def build_prompt(self, query: str, organized_chunks: Dict, context: QueryContext) -> str:
        """
        Build a context-aware prompt for the LLM.
        
        Args:
            query: Original user query
            organized_chunks: Organized chunks dictionary
            context: QueryContext with intent and concepts
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add intent-specific instructions
        if context.intent == 'location':
            prompt_parts.append("The user is asking WHERE to find something. Focus on file paths and locations.")
        elif context.intent == 'usage':
            prompt_parts.append("The user is asking HOW TO USE something. Focus on examples, function calls, and integration patterns.")
        elif context.intent == 'architecture':
            prompt_parts.append("The user is asking HOW something WORKS. Provide a comprehensive explanation of the architecture, flow, and implementation.")
        
        # Add organized context
        prompt_parts.append(f"\nMain concept: {context.main_concept}")
        if context.related_concepts:
            prompt_parts.append(f"Related concepts: {', '.join(context.related_concepts[:5])}")
        
        # Add chunks organized by file type
        by_file_type = organized_chunks.get('by_file_type', {})
        if by_file_type:
            prompt_parts.append("\nCode organized by file type:")
            for file_type, chunks in by_file_type.items():
                if chunks:
                    prompt_parts.append(f"\n--- {file_type.upper()} FILES ---")
                    for i, chunk in enumerate(chunks[:3], 1):  # Limit to 3 per type
                        file_path = chunk['metadata']['file']
                        lines = chunk['metadata']['lines']
                        code = chunk['code']
                        prompt_parts.append(f"\n{file_path} (lines {lines}):\n{code}\n")
        
        # Add chunks organized by content type
        by_content = organized_chunks.get('by_content', {})
        if by_content:
            prompt_parts.append("\nCode organized by content type:")
            for content_type, chunks in by_content.items():
                if chunks and content_type != 'other':
                    prompt_parts.append(f"\n--- {content_type.upper()} ---")
                    for i, chunk in enumerate(chunks[:2], 1):  # Limit to 2 per type
                        file_path = chunk['metadata']['file']
                        lines = chunk['metadata']['lines']
                        code = chunk['code']
                        prompt_parts.append(f"\n{file_path} (lines {lines}):\n{code}\n")
        
        return "\n".join(prompt_parts)

