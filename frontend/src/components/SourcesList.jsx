import { useState } from 'react';
import CodeBlock from './CodeBlock';

/**
 * SourcesList component - Show source files/lines for each answer
 */
export default function SourcesList({ sources, repoUrl }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandedSource, setExpandedSource] = useState(null);

  if (!sources || sources.length === 0) {
    return null;
  }

  const toggleSource = (index) => {
    setExpandedSource(expandedSource === index ? null : index);
  };

  // Construct GitHub URL for a source file
  const getGitHubUrl = (filePath, lines) => {
    if (!repoUrl) return null;
    
    try {
      // Clean up repo URL (remove .git, trailing slashes)
      let cleanRepoUrl = repoUrl.trim();
      
      // Handle different GitHub URL formats
      // https://github.com/user/repo.git -> https://github.com/user/repo
      // https://github.com/user/repo -> https://github.com/user/repo
      // git@github.com:user/repo.git -> https://github.com/user/repo
      if (cleanRepoUrl.startsWith('git@')) {
        // Convert SSH format to HTTPS
        cleanRepoUrl = cleanRepoUrl.replace('git@github.com:', 'https://github.com/');
      }
      
      if (cleanRepoUrl.endsWith('.git')) {
        cleanRepoUrl = cleanRepoUrl.slice(0, -4);
      }
      cleanRepoUrl = cleanRepoUrl.replace(/\/$/, '');
      
      // Validate it's a GitHub URL
      if (!cleanRepoUrl.includes('github.com')) {
        return null;
      }
      
      // Extract branch (default to main/master)
      // For now, we'll use 'main' - in a real app, you might want to track the branch
      const branch = 'main';
      
      // Construct GitHub blob URL
      const githubUrl = `${cleanRepoUrl}/blob/${branch}/${filePath}`;
      
      // Add line numbers if available
      if (lines) {
        const lineMatch = lines.match(/(\d+)-(\d+)/);
        if (lineMatch) {
          const startLine = lineMatch[1];
          const endLine = lineMatch[2];
          return `${githubUrl}#L${startLine}-L${endLine}`;
        } else {
          const singleLine = lines.match(/(\d+)/);
          if (singleLine) {
            return `${githubUrl}#L${singleLine[1]}`;
          }
        }
      }
      
      return githubUrl;
    } catch (error) {
      console.error('Error constructing GitHub URL:', error);
      return null;
    }
  };

  const previewCode = (code) => {
    if (!code) return '';
    return code.length > 300 ? code.substring(0, 300) + '...' : code;
  };

  return (
    <div className="mt-4 transition-all">
      {/* Collapsible button */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600 transition-all group"
      >
        <div className="flex items-center gap-2">
          <svg 
            className={`w-4 h-4 text-gray-600 dark:text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
          <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">
            View Sources ({sources.length})
          </span>
        </div>
        <svg className="w-4 h-4 text-gray-500 dark:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      </button>

      {/* Expanded sources */}
      {isExpanded && (
        <div className="mt-3 space-y-3 transition-all animate-in fade-in duration-200">
          {sources.map((source, index) => {
            const isSourceExpanded = expandedSource === index;
            const githubUrl = getGitHubUrl(source.file, source.lines);
            
            return (
              <div 
                key={index} 
                className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 shadow-sm transition-all hover:shadow-md"
              >
                {/* Source header */}
                <div className="flex items-start justify-between gap-3 mb-3">
                  <div className="flex items-start gap-3 flex-1 min-w-0">
                    {/* Numbered badge */}
                    <div className="flex-shrink-0 w-8 h-8 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded-full flex items-center justify-center text-sm font-bold">
                      {index + 1}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      {/* File path */}
                      <div className="font-mono text-sm font-medium text-gray-800 dark:text-gray-200 break-all mb-1">
                        {source.file}
                      </div>
                      {/* Line numbers */}
                      {source.lines && (
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          Lines {source.lines}
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {/* Action buttons */}
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {githubUrl && (
                      <a
                        href={githubUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded transition-all"
                        aria-label={`View ${source.file} on GitHub`}
                        title="View on GitHub"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                        </svg>
                      </a>
                    )}
                    {source.code && (
                      <button
                        onClick={() => toggleSource(index)}
                        className="p-2 text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded transition-all"
                        aria-expanded={isSourceExpanded}
                        aria-label={`${isSourceExpanded ? 'Hide' : 'Show'} code preview`}
                        title={isSourceExpanded ? 'Hide code' : 'Show code'}
                      >
                        <svg 
                          className={`w-4 h-4 transition-transform ${isSourceExpanded ? 'rotate-180' : ''}`} 
                          fill="none" 
                          stroke="currentColor" 
                          viewBox="0 0 24 24"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                      </button>
                    )}
                  </div>
                </div>
                
                {/* Expandable code preview */}
                {isSourceExpanded && source.code && (
                  <div className="mt-3 transition-all animate-in fade-in duration-200">
                    <CodeBlock
                      code={previewCode(source.code)}
                      language={source.language || 'text'}
                      fileName={source.file}
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
