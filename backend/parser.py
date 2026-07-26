"""
Code parser module for extracting and chunking code from GitHub repositories.
"""

import ast
import os
import re
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import git

# Repository size limits to prevent timeouts and excessive costs
MAX_FILES = 1000  # Maximum number of files to parse
MAX_CHUNKS = 5000  # Maximum number of code chunks to generate
MAX_REPO_SIZE_MB = 200  # Maximum repository size in MB

# Tree-sitter imports (optional - will fallback if not available)
try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("Warning: tree-sitter not available. Only Python files will be parsed intelligently.")


# Supported code file extensions
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.h', '.hpp', '.cc', '.cxx',
    '.go', '.rs', '.rb', '.php', '.cs', '.swift', '.kt', '.kts', '.scala',
    # Docs and config — often exactly what "what does this repo do?" needs
    '.md', '.rst', '.yaml', '.yml', '.toml', '.json',
}

# Priority files that should always be included and ranked higher
PRIORITY_FILES = {
    'README.md', 'readme.md', 'README.txt',
    'main.py', 'app.py', '__init__.py', '__main__.py',
    'index.js', 'index.ts', 'server.js', 'app.js',
    'App.jsx', 'App.tsx', 'index.jsx', 'index.tsx',
    'Main.java', 'Application.java',
    'main.cpp', 'main.c', 'main.go'
}

# Language mapping for tree-sitter
LANGUAGE_MAP = {
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.java': 'java',
    '.cpp': 'cpp',
    '.c': 'cpp',
    '.h': 'cpp',
    '.hpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.go': 'go',
    '.rs': 'rust',
    '.rb': 'ruby',
    '.php': 'php',
    '.cs': 'csharp',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.kts': 'kotlin',
    '.scala': 'scala',
}


def _get_layer(file_path: str) -> str:
    """Classify a file into backend / frontend / config / other."""
    file_lower = file_path.lower()
    ext = Path(file_path).suffix.lower()
    if ext in {'.jsx', '.tsx'} or 'frontend/' in file_lower or '/components/' in file_lower:
        return 'frontend'
    if ext == '.py':
        return 'backend'
    if ext in {'.json', '.yaml', '.yml', '.toml', '.ini'}:
        return 'config'
    if ext in {'.js', '.ts'}:
        return 'frontend' if ('frontend/' in file_lower or '/src/' in file_lower) else 'backend'
    return 'other'


def _generate_file_summary(file_path: str, repo_path: str) -> Optional[Dict[str, str]]:
    """Generate a synthetic summary chunk listing functions/classes with their docstrings."""
    ext = Path(file_path).suffix.lower()
    if ext not in {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h'}:
        return None

    full_path = Path(repo_path) / file_path
    summary_parts = [f"File: {file_path}"]

    if ext == '.py':
        try:
            source = full_path.read_text(encoding='utf-8')
            tree = ast.parse(source)
        except Exception:
            return None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                doc = (ast.get_docstring(node) or '').split('\n')[0]
                summary_parts.append(f"class {node.name}: {doc}")
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        cdoc = (ast.get_docstring(child) or '').split('\n')[0]
                        summary_parts.append(f"  def {child.name}: {cdoc}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = (ast.get_docstring(node) or '').split('\n')[0]
                summary_parts.append(f"def {node.name}: {doc}")
    else:
        try:
            source = full_path.read_text(encoding='utf-8')
        except Exception:
            return None
        for m in re.finditer(
            r'(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+(\w+)|class\s+(\w+)',
            source
        ):
            name = m.group(1) or m.group(2)
            kind = 'function' if m.group(1) else 'class'
            summary_parts.append(f"{kind} {name}")

    if len(summary_parts) <= 1:
        return None

    return {
        'code': '\n'.join(summary_parts),
        'file': file_path,
        'type': 'file_summary',
        'name': f"{Path(file_path).stem}_summary",
        'lines': '1-1',
    }


def clone_repo(repo_url: str, temp_dir: Optional[str] = None) -> str:
    """
    Clone a GitHub repository to a temporary directory.
    
    Args:
        repo_url: GitHub repository URL (e.g., 'https://github.com/user/repo.git')
        temp_dir: Optional temporary directory path. If None, creates a new temp directory.
    
    Returns:
        Path to the cloned repository directory
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix='repo_')
    else:
        os.makedirs(temp_dir, exist_ok=True)
    
    try:
        git.Repo.clone_from(repo_url, temp_dir)
        return temp_dir
    except git.exc.GitCommandError as e:
        raise ValueError(f"Failed to clone repository: {e}")


def find_code_files(repo_path: str) -> List[str]:
    """
    Find all code files in the repository based on file extensions.
    Limits the number of files to prevent timeouts on large repositories.
    
    Args:
        repo_path: Path to the repository root directory
    
    Returns:
        List of file paths relative to repo_path (limited to MAX_FILES)
    
    Raises:
        ValueError: If repository exceeds size limits
    """
    code_files = []
    repo_path_obj = Path(repo_path)
    
    # Check repository size
    total_size = 0
    for file_path in repo_path_obj.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    total_size_mb = total_size / (1024 * 1024)
    if total_size_mb > MAX_REPO_SIZE_MB:
        raise ValueError(
            f"Repository too large: {total_size_mb:.1f}MB (max {MAX_REPO_SIZE_MB}MB). "
            f"Please try a smaller repository."
        )
    
    # Directories to skip. Note: 'lib', 'libs', 'packages', and 'docs' are NOT
    # skipped — monorepos keep real source in packages/, many projects in lib/,
    # and docs are useful retrieval targets for a documentation assistant.
    skip_dirs = {
        '.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env',
        'dist', 'build', 'target', 'out', 'bin', 'obj',
        'vendor', 'third_party',
        'javadoc', 'apidocs',
        'test', 'tests', '__tests__', 'spec', 'specs',
        'coverage', '.pytest_cache', '.tox',
        'bower_components', 'jspm_packages',
        '.next', '.nuxt', '.cache', 'public/build',
        'site-packages', 'egg-info'
    }

    # File patterns to skip — precise suffixes/names only. Bare substrings like
    # 'bundle' or 'seed' skip legitimate source files (e.g. seedData.js).
    skip_file_patterns = [
        '.min.js', '.min.css', '.min.',   # Minified files
        '.bundle.js', '.chunk.js',        # Bundled files
        '.generated.', '.g.dart',         # Generated files
        '_pb2.py', '.pb.go',              # Protocol buffers
        '.test.', '.spec.',               # Test files
        'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',  # Lock files
        'poetry.lock', 'cargo.lock', 'composer.lock', 'gemfile.lock',
    ]
    
    for file_path in repo_path_obj.rglob('*'):
        # Stop if we've reached the file limit
        if len(code_files) >= MAX_FILES:
            print(f"⚠ Warning: Reached maximum file limit ({MAX_FILES} files). Stopping file discovery.")
            break
        
        # Skip directories
        if file_path.is_dir():
            continue
        
        # Skip files in ignored directories
        if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
            continue
        
        # Include if it's a supported code extension OR a priority file (e.g. README.md)
        is_code_file = file_path.suffix in CODE_EXTENSIONS
        is_priority = file_path.name in PRIORITY_FILES
        if not is_code_file and not is_priority:
            continue

        rel_path = file_path.relative_to(repo_path_obj)
        rel_path_str = str(rel_path)

        full_path = str(file_path)
        if should_skip_file(full_path, skip_file_patterns) or should_skip_file(rel_path_str, skip_file_patterns):
            continue

        code_files.append(rel_path_str)
    
    return sorted(code_files)


def should_skip_file(file_path: str, skip_file_patterns: List[str]) -> bool:
    """
    Check if a file should be skipped based on patterns.
    Returns True if file should be skipped.
    """
    file_path_lower = file_path.lower()
    
    # Skip test files
    if '/test/' in file_path_lower or '/tests/' in file_path_lower:
        return True

    # Skip if matches any pattern
    for pattern in skip_file_patterns:
        if pattern.lower() in file_path_lower:
            return True
    
    # Skip very large files (likely minified or generated)
    try:
        full_path = Path(file_path) if Path(file_path).is_absolute() else file_path
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            # Skip files larger than 500KB (likely minified libraries)
            if file_size > 500_000:
                return True
    except:
        pass
    
    return False


def chunk_python_file(file_path: str, repo_path: str) -> List[Dict[str, str]]:
    """
    Parse a Python file and chunk it by functions and classes.
    
    Args:
        file_path: Relative path to the Python file from repo_path
        repo_path: Path to the repository root directory
    
    Returns:
        List of dictionaries, each containing:
        - code: The code chunk (function or class)
        - file: Relative file path
        - type: 'function', 'class', or 'module'
        - name: Name of the function/class, or 'module' for top-level code
        - lines: Line range as string (e.g., "10-25")
    """
    full_path = Path(repo_path) / file_path
    chunks = []
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
            lines = source_code.split('\n')
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return chunks
    
    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        # Return the entire file as a single chunk if parsing fails
        return [{
            'code': source_code,
            'file': file_path,
            'type': 'module',
            'name': 'module',
            'lines': f"1-{len(lines)}"
        }]
    
    # Extract module-level constants/assignments (not inside any function or class)
    top_level_assign_lines = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            start = node.lineno
            end = getattr(node, 'end_lineno', start)
            top_level_assign_lines.extend(range(start, end + 1))

    if top_level_assign_lines:
        min_line = min(top_level_assign_lines)
        max_line = max(top_level_assign_lines)
        const_code = '\n'.join(lines[min_line - 1:max_line])
        if const_code.strip():
            chunks.append({
                'code': const_code,
                'file': file_path,
                'type': 'constants',
                'name': 'module_constants',
                'lines': f"{min_line}-{max_line}"
            })

    # Track line numbers for each node
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Get the actual source lines for this node
            # Start from the first decorator line so route paths (@app.get, @login_required, etc.) are included
            start_line = node.lineno
            if hasattr(node, 'decorator_list') and node.decorator_list:
                start_line = node.decorator_list[0].lineno
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line

            # Extract the code chunk
            chunk_lines = lines[start_line - 1:end_line]
            chunk_code = '\n'.join(chunk_lines)

            # Determine type
            if isinstance(node, ast.ClassDef):
                node_type = 'class'
            else:
                node_type = 'function'

            chunks.append({
                'code': chunk_code,
                'file': file_path,
                'type': node_type,
                'name': node.name,
                'lines': f"{start_line}-{end_line}"
            })

    # If no functions or classes found, return the entire file as a module chunk
    if not chunks:
        chunks.append({
            'code': source_code,
            'file': file_path,
            'type': 'module',
            'name': 'module',
            'lines': f"1-{len(lines)}"
        })
    
    return chunks


def chunk_with_tree_sitter(file_path: str, repo_path: str, language: str) -> List[Dict[str, str]]:
    """
    Parse a file using tree-sitter for languages other than Python.
    
    Args:
        file_path: Relative path to the file from repo_path
        language: Language name ('javascript', 'typescript', 'java', 'cpp')
        repo_path: Path to the repository root directory
    
    Returns:
        List of code chunks (functions, classes, etc.)
    """
    if not TREE_SITTER_AVAILABLE:
        return []
    
    full_path = Path(repo_path) / file_path
    chunks = []
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
            lines = source_code.split('\n')
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return chunks
    
    try:
        # Try to load language (this requires compiled language libraries)
        # For now, we'll do basic parsing without compiled grammars
        # In production, you'd compile the grammars first
        parser = Parser()
        
        # Note: This is a simplified version. In production, you need to:
        # 1. Build language grammars: tree-sitter build --output languages.so languages/
        # 2. Load them: Language('languages.so', language)
        
        # For now, fallback to simple function/class extraction using regex/patterns
        return chunk_by_patterns(source_code, file_path, language, lines)
        
    except Exception as e:
        print(f"Error parsing {file_path} with tree-sitter: {e}")
        # Fallback to pattern-based chunking
        return chunk_by_patterns(source_code, file_path, language, lines)


# Whole-file fallback chunks are split into windows so a single large file
# doesn't exceed the embedding model's token limit (~8K tokens).
_WINDOW_LINES = 120


def _windowed_chunks(source_code: str, file_path: str, chunk_type: str = 'file') -> List[Dict[str, str]]:
    """Split a file into fixed-size line windows (one chunk for small files)."""
    lines = source_code.split('\n')
    stem = Path(file_path).stem
    if len(lines) <= _WINDOW_LINES:
        return [{
            'code': source_code,
            'file': file_path,
            'type': chunk_type,
            'name': stem,
            'lines': f"1-{len(lines)}",
        }]
    chunks = []
    for start in range(0, len(lines), _WINDOW_LINES):
        window = lines[start:start + _WINDOW_LINES]
        chunks.append({
            'code': '\n'.join(window),
            'file': file_path,
            'type': chunk_type,
            'name': f"{stem}_part{start // _WINDOW_LINES + 1}",
            'lines': f"{start + 1}-{start + len(window)}",
        })
    return chunks


def chunk_by_patterns(source_code: str, file_path: str, language: str, lines: List[str]) -> List[Dict[str, str]]:
    """
    Fallback chunking using regex patterns for functions and classes.
    Works for JavaScript, TypeScript, Java, C++.
    """
    import re
    chunks = []
    
    # Patterns for different languages
    patterns = {
        'javascript': {
            'function': r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{',
            'class': r'(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{',
            'method': r'(\w+)\s*\([^)]*\)\s*\{',
        },
        'typescript': {
            'function': r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)(?::\s*\w+)?\s*\{',
            'class': r'(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{',
            'method': r'(\w+)\s*\([^)]*\)(?::\s*\w+)?\s*\{',
        },
        'java': {
            'class': r'(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)',
            'method': r'(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*\{',
        },
        'cpp': {
            'class': r'class\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+\w+)?\s*\{',
            'function': r'(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*\{',
        },
    }
    
    lang_patterns = patterns.get(language, {})
    if not lang_patterns:
        # No patterns for this language — fall back to windowed whole-file chunks
        return _windowed_chunks(source_code, file_path)
    
    # Find classes first
    for pattern_type, pattern in lang_patterns.items():
        if pattern_type == 'class':
            for match in re.finditer(pattern, source_code, re.MULTILINE):
                name = match.group(1)
                start_pos = match.start()
                start_line = source_code[:start_pos].count('\n') + 1
                
                # Find matching closing brace (simplified)
                brace_count = 0
                end_pos = start_pos
                for i, char in enumerate(source_code[start_pos:], start_pos):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                end_line = source_code[:end_pos].count('\n') + 1
                chunk_code = source_code[start_pos:end_pos]
                
                chunks.append({
                    'code': chunk_code,
                    'file': file_path,
                    'type': 'class',
                    'name': name,
                    'lines': f"{start_line}-{end_line}"
                })
    
    # Find functions/methods
    for pattern_type, pattern in lang_patterns.items():
        if pattern_type in ['function', 'method']:
            for match in re.finditer(pattern, source_code, re.MULTILINE):
                name = match.group(1)
                start_pos = match.start()
                start_line = source_code[:start_pos].count('\n') + 1
                
                # Skip if already in a class chunk
                in_class = False
                for chunk in chunks:
                    if chunk['type'] == 'class':
                        chunk_start = int(chunk['lines'].split('-')[0])
                        chunk_end = int(chunk['lines'].split('-')[1])
                        if chunk_start <= start_line <= chunk_end:
                            in_class = True
                            break
                
                if in_class:
                    continue
                
                # Find matching closing brace
                brace_count = 0
                end_pos = start_pos
                for i, char in enumerate(source_code[start_pos:], start_pos):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                end_line = source_code[:end_pos].count('\n') + 1
                chunk_code = source_code[start_pos:end_pos]
                
                chunks.append({
                    'code': chunk_code,
                    'file': file_path,
                    'type': 'function',
                    'name': name,
                    'lines': f"{start_line}-{end_line}"
                })
    
    # If no chunks found, return the file in windows
    if not chunks:
        chunks = _windowed_chunks(source_code, file_path)

    return chunks


def _parse_single_file(file_path: str, repo_path: str) -> List[Dict[str, str]]:
    """Parse a single file into code chunks. Used by parallel parser."""
    file_ext = Path(file_path).suffix
    layer = _get_layer(file_path)
    chunks: List[Dict[str, str]] = []
    try:
        if file_ext == '.py':
            chunks = chunk_python_file(file_path, repo_path)
        elif file_ext in LANGUAGE_MAP:
            language = LANGUAGE_MAP[file_ext]
            chunks = chunk_with_tree_sitter(file_path, repo_path, language)
            if not chunks:
                full_path = Path(repo_path) / file_path
                source_code = full_path.read_text(encoding='utf-8')
                chunks = _windowed_chunks(source_code, file_path)
        else:
            full_path = Path(repo_path) / file_path
            source_code = full_path.read_text(encoding='utf-8')
            chunk_type = 'documentation' if file_ext in ('.md', '.rst') else 'file'
            chunks = _windowed_chunks(source_code, file_path, chunk_type)

        for chunk in chunks:
            chunk['layer'] = layer

        summary = _generate_file_summary(file_path, repo_path)
        if summary:
            summary['layer'] = layer
            chunks.append(summary)

        return chunks
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


def parse_repo(repo_url: str, temp_dir: Optional[str] = None, cleanup: bool = True) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """
    Clone a repository and parse all code files into chunks.
    
    Currently only supports Python files. Other languages will be added later.
    
    Args:
        repo_url: GitHub repository URL
        temp_dir: Optional temporary directory for cloning. If None, creates temp dir.
        cleanup: Whether to cleanup temporary directory after parsing (default: True)
    
    Returns:
        Tuple of (list of code chunks, temp_dir_path)
        - chunks: List of dictionaries with code, file, type, name, lines
        - temp_dir_path: Path to temp directory (None if cleaned up)
    """
    repo_path = None
    created_temp_dir = temp_dir is None
    
    try:
        # Clone the repository
        repo_path = clone_repo(repo_url, temp_dir)
        
        # Find all code files
        code_files = find_code_files(repo_path)
        
        print(f"Found {len(code_files)} code files to parse...")

        # Parse files in parallel — I/O and CPU-bound parsing benefit from threads here
        from concurrent.futures import ThreadPoolExecutor
        all_chunks = []
        workers = min(8, max(1, len(code_files)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(
                lambda fp: _parse_single_file(fp, repo_path),
                code_files
            ))

        for chunks in results:
            all_chunks.extend(chunks)
            if len(all_chunks) >= MAX_CHUNKS:
                print(f"⚠ Warning: Reached maximum chunk limit ({MAX_CHUNKS} chunks). Truncating.")
                all_chunks = all_chunks[:MAX_CHUNKS]
                break
        
        print(f"✓ Parsed {len(all_chunks)} code chunks from {len(code_files)} files")
        
        # Prioritize important files
        all_chunks = prioritize_chunks(all_chunks)
        
        return all_chunks, repo_path if not cleanup else None
        
    except Exception as e:
        # Ensure cleanup even on error
        if cleanup and created_temp_dir and repo_path and os.path.exists(repo_path):
            try:
                shutil.rmtree(repo_path)
                print(f"✓ Cleaned up temporary directory after error: {repo_path}")
            except Exception as cleanup_error:
                print(f"⚠ Warning: Failed to cleanup temporary directory {repo_path}: {cleanup_error}")
        raise  # Re-raise the original error
    finally:
        # Cleanup temporary directory if we created it and cleanup is requested
        if cleanup and created_temp_dir and repo_path and os.path.exists(repo_path):
            try:
                shutil.rmtree(repo_path)
                print(f"✓ Cleaned up temporary directory: {repo_path}")
            except Exception as e:
                print(f"⚠ Warning: Failed to cleanup temporary directory {repo_path}: {e}")
                # Don't raise - we've already parsed the repo successfully


def prioritize_chunks(chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Prioritize chunks so important files appear first in search results.
    This helps with queries like "what does this project do?"
    """
    priority = []
    normal = []
    
    for chunk in chunks:
        filename = os.path.basename(chunk['file'])
        if filename in PRIORITY_FILES:
            priority.append(chunk)
        else:
            normal.append(chunk)
    
    return priority + normal


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parser.py <github_repo_url>")
        sys.exit(1)
    
    repo_url = sys.argv[1]
    chunks = parse_repo(repo_url)
    
    print(f"Found {len(chunks)} code chunks")
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        print(f"\nChunk {i+1}:")
        print(f"  File: {chunk['file']}")
        print(f"  Type: {chunk['type']}")
        print(f"  Name: {chunk['name']}")
        print(f"  Lines: {chunk['lines']}")
        print(f"  Code preview: {chunk['code'][:100]}...")

