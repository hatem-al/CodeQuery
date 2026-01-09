"""
Code parser module for extracting and chunking code from GitHub repositories.
"""

import ast
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import git

# Tree-sitter imports (optional - will fallback if not available)
try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("Warning: tree-sitter not available. Only Python files will be parsed intelligently.")


# Supported code file extensions
CODE_EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.h', '.hpp', '.cc', '.cxx'}

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
    
    Args:
        repo_path: Path to the repository root directory
    
    Returns:
        List of file paths relative to repo_path
    """
    code_files = []
    repo_path_obj = Path(repo_path)
    
    # Directories to skip
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env', 'dist', 'build'}
    
    for file_path in repo_path_obj.rglob('*'):
        # Skip directories
        if file_path.is_dir():
            continue
        
        # Skip files in ignored directories
        if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
            continue
        
        # Check if file has a supported code extension
        if file_path.suffix in CODE_EXTENSIONS:
            # Get relative path from repo root
            rel_path = file_path.relative_to(repo_path_obj)
            code_files.append(str(rel_path))
    
    return sorted(code_files)


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
    
    # Track line numbers for each node
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Get the actual source lines for this node
            start_line = node.lineno
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
        # If no patterns, return entire file as one chunk
        return [{
            'code': source_code,
            'file': file_path,
            'type': 'file',
            'name': Path(file_path).stem,
            'lines': f"1-{len(lines)}"
        }]
    
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
    
    # If no chunks found, return entire file
    if not chunks:
        chunks.append({
            'code': source_code,
            'file': file_path,
            'type': 'file',
            'name': Path(file_path).stem,
            'lines': f"1-{len(lines)}"
        })
    
    return chunks


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
        
        # Parse and chunk files
        all_chunks = []
        
        for file_path in code_files:
            file_ext = Path(file_path).suffix
            
            try:
                if file_ext == '.py':
                    # Parse Python files using AST
                    chunks = chunk_python_file(file_path, repo_path)
                    all_chunks.extend(chunks)
                elif file_ext in LANGUAGE_MAP:
                    # Parse other languages using tree-sitter or pattern matching
                    language = LANGUAGE_MAP[file_ext]
                    chunks = chunk_with_tree_sitter(file_path, repo_path, language)
                    if chunks:
                        all_chunks.extend(chunks)
                    else:
                        # Fallback to entire file if parsing fails
                        full_path = Path(repo_path) / file_path
                        try:
                            with open(full_path, 'r', encoding='utf-8') as f:
                                source_code = f.read()
                                lines = source_code.split('\n')
                            
                            all_chunks.append({
                                'code': source_code,
                                'file': file_path,
                                'type': 'file',
                                'name': Path(file_path).stem,
                                'lines': f"1-{len(lines)}"
                            })
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
                            continue
                else:
                    # Unknown file type - add as single chunk
                    full_path = Path(repo_path) / file_path
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            source_code = f.read()
                            lines = source_code.split('\n')
                        
                        all_chunks.append({
                            'code': source_code,
                            'file': file_path,
                            'type': 'file',
                            'name': Path(file_path).stem,
                            'lines': f"1-{len(lines)}"
                        })
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
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

