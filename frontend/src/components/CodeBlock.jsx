import { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

/**
 * CodeBlock component - Syntax-highlighted code snippets with copy button
 */
export default function CodeBlock({ code, language = 'text', fileName }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      // Copy only the code content, not line numbers
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = code;
      textArea.style.position = 'fixed';
      textArea.style.opacity = '0';
      document.body.appendChild(textArea);
      textArea.select();
      try {
        document.execCommand('copy');
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (fallbackErr) {
        console.error('Fallback copy failed:', fallbackErr);
      }
      document.body.removeChild(textArea);
    }
  };

  // Detect language from file extension if not provided
  const detectLanguage = (lang, fileName) => {
    if (lang && lang !== 'unknown' && lang !== 'text') return lang;
    
    if (fileName) {
      const ext = fileName.split('.').pop()?.toLowerCase();
      const langMap = {
        'js': 'javascript',
        'jsx': 'jsx',
        'ts': 'typescript',
        'tsx': 'tsx',
        'py': 'python',
        'java': 'java',
        'cpp': 'cpp',
        'c': 'c',
        'h': 'c',
        'hpp': 'cpp',
        'html': 'html',
        'css': 'css',
        'json': 'json',
        'md': 'markdown',
        'sh': 'bash',
        'yml': 'yaml',
        'yaml': 'yaml',
        'go': 'go',
        'rs': 'rust',
        'rb': 'ruby',
        'php': 'php',
        'sql': 'sql',
        'xml': 'xml'
      };
      return langMap[ext] || 'text';
    }
    
    return lang || 'text';
  };

  const detectedLang = detectLanguage(language, fileName);
  const displayLang = detectedLang.toUpperCase();

  return (
    <div className="relative group my-2 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 transition-all">
      {/* Header with language badge and copy button */}
      <div className="flex items-center justify-between bg-gray-800 px-4 py-2.5">
        {/* Language badge - top left */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-gray-300 uppercase tracking-wider px-2 py-1 bg-gray-700 rounded">
            {displayLang}
          </span>
        </div>
        
        {/* Copy button - top right */}
        <button
          onClick={handleCopy}
          className={`ml-2 px-3 py-1.5 text-xs rounded transition-all flex items-center gap-1.5 flex-shrink-0 min-h-[32px] ${
            copied 
              ? 'bg-green-600 text-white' 
              : 'bg-blue-600 hover:bg-blue-700 text-white'
          }`}
          title="Copy code"
          aria-label="Copy code to clipboard"
        >
          {copied ? (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <span className="hidden sm:inline">Copied!</span>
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <span className="hidden sm:inline">Copy</span>
            </>
          )}
        </button>
      </div>
      
      {/* Code content */}
      <div className="overflow-x-auto">
        <SyntaxHighlighter
          language={detectedLang}
          style={vscDarkPlus}
          customStyle={{
            margin: 0,
            borderRadius: 0,
            fontSize: '0.875rem',
            padding: '1rem',
            overflow: 'visible',
            backgroundColor: '#1e1e1e'
          }}
          showLineNumbers={code.split('\n').length > 1}
          wrapLines={true}
          wrapLongLines={true}
          PreTag="div"
          lineNumberStyle={{
            color: '#6b7280',
            paddingRight: '1rem',
            paddingLeft: '0.5rem',
            minWidth: '2.5rem',
            userSelect: 'none'
          }}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
