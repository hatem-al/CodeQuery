"""
Query processing utilities for typo detection and correction.
"""

from typing import Tuple, List, Dict

# Common typo mappings (typo -> correct)
COMMON_TYPOS = {
    'middlewhere': 'middleware',
    'authentification': 'authentication',
    'databse': 'database',
    'databasse': 'database',
    'databas': 'database',
    'middlware': 'middleware',
    'middlewaree': 'middleware',
    'authenticaton': 'authentication',
    'authenticaiton': 'authentication',
    'authnetication': 'authentication',
    'authenication': 'authentication',
    'endpoit': 'endpoint',
    'endpoits': 'endpoints',
    'endpont': 'endpoint',
    'endponts': 'endpoints',
    'componet': 'component',
    'componets': 'components',
    'componenet': 'component',
    'componenets': 'components',
    'functon': 'function',
    'functons': 'functions',
    'functoin': 'function',
    'functoins': 'functions',
    'varable': 'variable',
    'varables': 'variables',
    'varaiable': 'variable',
    'varaiables': 'variables',
    'configuraton': 'configuration',
    'configuratoin': 'configuration',
    'configuartion': 'configuration',
    'validaton': 'validation',
    'validatoin': 'validation',
    'validaiton': 'validation',
    'handlng': 'handling',
    'handilng': 'handling',
    'handlnig': 'handling',
    'routng': 'routing',
    'routnig': 'routing',
    'routin': 'routing',
    'requst': 'request',
    'requets': 'requests',
    'requets': 'requests',
    'responce': 'response',
    'respones': 'response',
    'responce': 'response',
    'eror': 'error',
    'erors': 'errors',
    'erorr': 'error',
    'erorrs': 'errors',
    'excepton': 'exception',
    'exceptons': 'exceptions',
    'exceptoin': 'exception',
    'exceptoins': 'exceptions',
}


def fix_common_typos(query: str) -> Tuple[str, bool, List[str]]:
    """
    Fix common typos in a query string.
    
    Args:
        query: The original query string
        
    Returns:
        Tuple of (corrected_query, was_corrected, corrections_made)
        - corrected_query: The query with typos fixed
        - was_corrected: Boolean indicating if any corrections were made
        - corrections_made: List of tuples (original, corrected) for corrections made
    """
    original_query = query
    corrections = []
    
    # Split query into words (preserving case)
    words = query.split()
    corrected_words = []
    
    for word in words:
        # Remove punctuation for comparison but keep it in the word
        word_lower = word.lower().strip('.,!?;:()[]{}"\'')
        original_word = word
        
        # Check if word (without punctuation) is a typo
        if word_lower in COMMON_TYPOS:
            corrected_word_lower = COMMON_TYPOS[word_lower]
            
            # Preserve original case pattern
            if word.isupper():
                corrected_word = corrected_word_lower.upper()
            elif word[0].isupper():
                corrected_word = corrected_word_lower.capitalize()
            else:
                corrected_word = corrected_word_lower
            
            # Preserve punctuation
            if not word[-1].isalnum():
                corrected_word = corrected_word + word[-1]
            if not word[0].isalnum():
                corrected_word = word[0] + corrected_word
            
            corrections.append((word_lower, corrected_word_lower))
            corrected_words.append(corrected_word)
        else:
            corrected_words.append(word)
    
    corrected_query = ' '.join(corrected_words)
    was_corrected = len(corrections) > 0
    
    return corrected_query, was_corrected, corrections


def suggest_alternatives(query: str, low_confidence_terms: List[str] = None) -> List[str]:
    """
    Suggest alternative query terms based on common patterns.
    
    Args:
        query: The original query
        low_confidence_terms: Terms that had low confidence in search results
        
    Returns:
        List of suggested alternative queries
    """
    suggestions = []
    
    # Common patterns for suggestions
    if 'middleware' in query.lower() or 'middlewhere' in query.lower():
        suggestions.extend([
            'authentication middleware',
            'error handling middleware',
            'request middleware',
            'middleware configuration'
        ])
    
    if 'authentication' in query.lower() or 'authentification' in query.lower():
        suggestions.extend([
            'authentication flow',
            'login authentication',
            'JWT authentication',
            'authentication middleware'
        ])
    
    if 'database' in query.lower() or 'databse' in query.lower():
        suggestions.extend([
            'database configuration',
            'database connection',
            'database schema',
            'database queries'
        ])
    
    if 'endpoint' in query.lower() or 'endpoit' in query.lower():
        suggestions.extend([
            'API endpoints',
            'endpoint handlers',
            'route endpoints',
            'REST endpoints'
        ])
    
    # Generic suggestions based on low confidence terms
    if low_confidence_terms:
        for term in low_confidence_terms[:3]:  # Limit to 3 suggestions
            if term not in query.lower():
                suggestions.append(f"{query} {term}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_suggestions = []
    for suggestion in suggestions:
        if suggestion.lower() not in seen:
            seen.add(suggestion.lower())
            unique_suggestions.append(suggestion)
    
    return unique_suggestions[:4]  # Return max 4 suggestions

