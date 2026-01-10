import { useState } from 'react';
import axios from 'axios';
import LoadingSpinner from './LoadingSpinner';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

/**
 * EmptyState component - Shows when no repositories are indexed
 */
export default function EmptyState({ onRepoIndexed }) {
  const [repoUrl, setRepoUrl] = useState('');
  const [isIndexing, setIsIndexing] = useState(false);
  const [error, setError] = useState(null);

  const pollIndexingStatus = async (repoUrl) => {
    const maxAttempts = 60;
    let attempts = 0;
    
    const poll = async () => {
      try {
        const encodedUrl = encodeURIComponent(repoUrl);
        const response = await axios.get(`${API_BASE_URL}/index/status/${encodedUrl}`);
        const status = response.data;
        
        if (status.status === 'completed') {
          if (onRepoIndexed) {
            onRepoIndexed(repoUrl);
          }
          setRepoUrl('');
          setIsIndexing(false);
          return true;
        } else if (status.status === 'error') {
          setError(status.error || 'Failed to index repository. Please try again.');
          setIsIndexing(false);
          return true;
        } else if (status.status === 'already_indexed') {
          if (onRepoIndexed) {
            onRepoIndexed(repoUrl);
          }
          setIsIndexing(false);
          return true;
        }
        
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 5000);
        } else {
          setError('Indexing is taking longer than expected. Please check back later.');
          setIsIndexing(false);
        }
      } catch (err) {
        if (err.response?.status === 404) {
          attempts++;
          if (attempts < maxAttempts) {
            setTimeout(poll, 5000);
          } else {
            setError('Failed to track indexing progress.');
            setIsIndexing(false);
          }
        } else {
          setError('Error checking indexing status.');
          setIsIndexing(false);
        }
      }
    };
    
    setTimeout(poll, 2000);
  };

  const handleIndex = async () => {
    if (!repoUrl.trim()) {
      setError('Please enter a GitHub repository URL');
      return;
    }

    // Validate GitHub URL
    const githubUrlPattern = /^https?:\/\/(www\.)?github\.com\/[\w\-\.]+\/[\w\-\.]+(\.git)?(\/.*)?$/;
    const trimmedUrl = repoUrl.trim();
    if (!githubUrlPattern.test(trimmedUrl)) {
      setError('Please enter a valid GitHub repository URL');
      return;
    }

    setIsIndexing(true);
    setError(null);

    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(
        `${API_BASE_URL}/index`,
        {
          repo_url: trimmedUrl,
          force_reindex: false
        },
        {
          headers: { Authorization: `Bearer ${token}` },
          timeout: 60000 // 60 seconds for cold start
        }
      );

      if (response.data.status === 'indexed' || response.data.status === 'already_indexed') {
        if (onRepoIndexed) {
          onRepoIndexed(response.data.repo_id);
        }
        setRepoUrl('');
        setIsIndexing(false);
      } else if (response.data.status === 'indexing') {
        pollIndexingStatus(trimmedUrl);
      }
    } catch (err) {
      let errorMessage = 'Failed to index repository. Please try again.';
      
      if (err.code === 'ERR_NETWORK' || err.message === 'Network Error') {
        errorMessage = 'Cannot connect to backend. Please check your connection.';
      } else if (err.response?.status === 404) {
        errorMessage = 'Repository not found or is private.';
      } else if (err.response?.status === 400 && err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      }
      
      setError(errorMessage);
      setIsIndexing(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] px-4">
      {/* Icon */}
      <div className="text-6xl mb-4">🔍</div>
      
      {/* Heading */}
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4 text-center">
        Search Any Codebase with AI
      </h1>
      
      {/* Subtitle */}
      <p className="text-lg text-gray-600 dark:text-gray-300 mb-8 text-center max-w-2xl">
        Index a GitHub repository and ask questions in natural language
      </p>

      {/* Repository Input */}
      <div className="w-full max-w-2xl mb-12">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          GitHub Repository URL
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={repoUrl}
            onChange={(e) => {
              setRepoUrl(e.target.value);
              setError(null);
            }}
            placeholder="https://github.com/username/repo"
            className="flex-1 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            disabled={isIndexing}
          />
          <button
            onClick={handleIndex}
            disabled={isIndexing || !repoUrl.trim()}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium shadow-md hover:shadow-lg flex items-center gap-2"
          >
            {isIndexing ? (
              <>
                <LoadingSpinner size="sm" />
                <span>Indexing...</span>
              </>
            ) : (
              <span>Index Repository</span>
            )}
          </button>
        </div>
        
        {error && (
          <div className="mt-2 p-3 bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-300 rounded-lg text-sm border border-red-200 dark:border-red-800">
            {error}
          </div>
        )}

        {isIndexing && (
          <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <p className="text-sm text-blue-800 dark:text-blue-300">
              Indexing repository... This may take a few minutes depending on repository size.
            </p>
          </div>
        )}
      </div>

      {/* Feature Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-4xl">
        {/* Card 1 */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-md border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-all">
          <div className="text-4xl mb-3">⚡</div>
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
            Instant Search
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            Semantic search across 50K+ lines of code in milliseconds
          </p>
        </div>

        {/* Card 2 */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-md border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-all">
          <div className="text-4xl mb-3">🤖</div>
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
            AI-Powered
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            GPT-4 explains code with examples and context
          </p>
        </div>

        {/* Card 3 */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-md border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-all">
          <div className="text-4xl mb-3">📚</div>
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
            Multi-Language
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            Python, JavaScript, Java, C++, and more
          </p>
        </div>
      </div>
    </div>
  );
}

