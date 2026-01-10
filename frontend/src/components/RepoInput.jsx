import { useState, useEffect } from 'react';
import axios from 'axios';
import LoadingSpinner from './LoadingSpinner';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

// Helper to get auth token
const getAuthToken = () => {
  return localStorage.getItem('token');
};

// Configure axios to include token in requests
axios.interceptors.request.use((config) => {
  const token = getAuthToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

/**
 * RepoInput component - Input GitHub URL and trigger indexing
 */
export default function RepoInput({ onRepoIndexed, currentRepo }) {
  const [repoUrl, setRepoUrl] = useState('');
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexStatus, setIndexStatus] = useState(null);
  const [error, setError] = useState(null);
  const [indexedRepos, setIndexedRepos] = useState([]);
  const [loadingRepos, setLoadingRepos] = useState(true);
  const [indexingStage, setIndexingStage] = useState('');

  // Load indexed repos on mount
  useEffect(() => {
    loadIndexedRepos();
  }, []);

  const loadIndexedRepos = async () => {
    try {
      setLoadingRepos(true);
      const response = await axios.get(`${API_BASE_URL}/repos`);
      setIndexedRepos(response.data.repos || []);
    } catch (err) {
      console.error('Error loading repos:', err);
      // Don't show error if backend is down on initial load
      if (err.code !== 'ERR_NETWORK') {
        setError('Failed to load repositories. Please refresh the page.');
      }
    } finally {
      setLoadingRepos(false);
    }
  };

  const pollIndexingStatus = async (repoUrl) => {
    const maxAttempts = 60; // Poll for up to 5 minutes (5 seconds * 60 = 5 min)
    let attempts = 0;
    
    const poll = async () => {
      try {
        const encodedUrl = encodeURIComponent(repoUrl);
        const response = await axios.get(`${API_BASE_URL}/index/status/${encodedUrl}`);
        const status = response.data;
        
        // Update stage
        setIndexingStage(status.stage || 'Indexing...');
        
        if (status.status === 'completed') {
          // Success!
          setIndexStatus({
            status: 'indexed',
            chunks_indexed: status.chunks_indexed
          });
          await loadIndexedRepos();
          if (onRepoIndexed) {
            onRepoIndexed(repoUrl);
          }
          setRepoUrl('');
          setIsIndexing(false);
          return true;
        } else if (status.status === 'error') {
          // Error
          setError(status.error || 'Failed to index repository. Please try again.');
          setIsIndexing(false);
          return true;
        } else if (status.status === 'already_indexed') {
          // Already indexed
          setIndexStatus({
            status: 'already_indexed',
            chunks_indexed: status.chunks_indexed
          });
          await loadIndexedRepos();
          setIsIndexing(false);
          return true;
        }
        
        // Still indexing, continue polling
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 5000); // Poll every 5 seconds
        } else {
          setError('Indexing is taking longer than expected. Please check back later.');
          setIsIndexing(false);
        }
      } catch (err) {
        if (err.response?.status === 404) {
          // No progress found yet, try again
          attempts++;
          if (attempts < maxAttempts) {
            setTimeout(poll, 5000);
          } else {
            setError('Failed to track indexing progress. Please check back later.');
            setIsIndexing(false);
          }
        } else {
          setError('Error checking indexing status. Please refresh the page.');
          setIsIndexing(false);
        }
      }
    };
    
    // Start polling after a short delay
    setTimeout(poll, 2000);
  };

  const handleIndex = async (forceReindex = false) => {
    if (!repoUrl.trim()) {
      setError('Please enter a GitHub repository URL');
      return;
    }

    // Validate GitHub URL - more flexible pattern
    const githubUrlPattern = /^https?:\/\/(www\.)?github\.com\/[\w\-\.]+\/[\w\-\.]+(\.git)?(\/.*)?$/;
    const trimmedUrl = repoUrl.trim();
    if (!githubUrlPattern.test(trimmedUrl)) {
      setError('Please enter a valid GitHub repository URL (e.g., https://github.com/user/repo)');
      return;
    }

    setIsIndexing(true);
    setError(null);
    setIndexStatus(null);
    setIndexingStage('Connecting to backend...');

    try {
      // Try to call the index endpoint with retry for cold starts
      let response;
      let attempts = 0;
      const maxAttempts = 2;
      
      while (attempts < maxAttempts) {
        try {
          setIndexingStage(attempts > 0 ? 'Waking up server (cold start)...' : 'Starting indexing...');
          response = await axios.post(`${API_BASE_URL}/index`, {
            repo_url: trimmedUrl,
            force_reindex: forceReindex
          }, {
            timeout: 60000 // 60 seconds for initial response
          });
          break; // Success, exit loop
        } catch (err) {
          attempts++;
          if (err.code === 'ECONNABORTED' && attempts < maxAttempts) {
            // Timeout, retry once
            setIndexingStage('Server is waking up, retrying...');
            await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
            continue;
          }
          throw err; // Re-throw if not a timeout or max attempts reached
        }
      }

      if (response.data.status === 'indexed' || response.data.status === 'already_indexed') {
        // Already indexed (synchronous response)
        setIndexStatus(response.data);
        await loadIndexedRepos();
        if (onRepoIndexed) {
          onRepoIndexed(response.data.repo_id);
        }
        setRepoUrl('');
        setIsIndexing(false);
      } else if (response.data.status === 'indexing') {
        // Background indexing started, poll for status
        setIndexingStage('Cloning repository...');
        pollIndexingStatus(trimmedUrl);
      }
    } catch (err) {
      let errorMessage = 'Failed to index repository. Please try again.';
      
      if (err.code === 'ECONNABORTED') {
        errorMessage = 'Server is taking too long to respond (cold start). Please try again in a moment.';
      } else if (err.code === 'ERR_NETWORK' || err.message === 'Network Error') {
        errorMessage = 'Cannot connect to backend. Please check your connection and try again.';
      } else if (err.response?.status === 404) {
        errorMessage = 'Repository not found or is private. Please check the URL and try again.';
      } else if (err.response?.status === 429) {
        errorMessage = 'OpenAI rate limit exceeded. Please wait a moment and try again.';
      } else if (err.response?.status === 400 && err.response?.data?.detail) {
        // Handle repository size limit errors
        errorMessage = err.response.data.detail;
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.message && err.message.includes('timeout')) {
        errorMessage = 'Request timed out. The server may be starting up (cold start). Please try again in a moment.';
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
      setIsIndexing(false);
      setIndexingStage('');
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!isIndexing && repoUrl.trim()) {
    handleIndex();
    }
  };

  const handleSelectRepo = (repoUrl) => {
    if (onRepoIndexed) {
      onRepoIndexed(repoUrl);
    }
  };

  const handleDeleteRepo = async (repoUrlToDelete) => {
    if (!window.confirm(`Are you sure you want to remove "${repoUrlToDelete}" from the indexed repositories? This will delete all indexed data for this repository.`)) {
      return;
    }
    
    try {
      // URL encode the repo URL for the API call
      const encodedUrl = encodeURIComponent(repoUrlToDelete);
      const response = await axios.delete(`${API_BASE_URL}/repos/${encodedUrl}`);
      
      if (response.data.status === 'deleted') {
        // Reload the repos list
        await loadIndexedRepos();
        
        // Clear current repo if it was deleted
        if (currentRepo === repoUrlToDelete && onRepoIndexed) {
          onRepoIndexed(null);
        }
        
        setIndexStatus({
          status: 'deleted',
          message: `Repository ${repoUrlToDelete} has been deleted`
        });
      }
    } catch (err) {
      setError(
        err.response?.data?.detail || 
        err.message || 
        'Failed to delete repository. Please try again.'
      );
    }
  };

  const formatRepoName = (url) => {
    try {
      const match = url.match(/github\.com\/([\w\-\.]+\/[\w\-\.]+)/);
      return match ? match[1] : url;
    } catch {
      return url;
    }
  };

  const isRepoAlreadyIndexed = indexedRepos.some(repo => repo.repo_url === repoUrl.trim());

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 sm:p-6 mb-6">
      <h2 className="text-xl sm:text-2xl font-bold text-gray-800 dark:text-white mb-4">Index Repository</h2>
      
      {/* Empty State */}
      {!loadingRepos && indexedRepos.length === 0 && !isIndexing && (
        <div className="text-center py-8 px-4">
          <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2h-2M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2" />
          </svg>
          <p className="text-gray-600 mb-2">No repositories indexed yet</p>
          <p className="text-sm text-gray-500">Index your first repository to get started</p>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="repo-url" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            GitHub Repository URL
          </label>
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              id="repo-url"
              type="text"
              value={repoUrl}
              onChange={(e) => {
                setRepoUrl(e.target.value);
                setError(null); // Clear error when user types
              }}
              placeholder="Enter GitHub repository URL (e.g., https://github.com/user/repo)"
              className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:cursor-not-allowed transition-colors bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
              disabled={isIndexing}
              aria-label="GitHub repository URL"
            />
            <button
              type="submit"
              disabled={isIndexing || !repoUrl.trim()}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium min-h-[44px] sm:min-h-0"
              aria-label="Index repository"
            >
              {isIndexing ? 'Indexing...' : 'Index Repository'}
            </button>
          </div>
        </div>

        {isRepoAlreadyIndexed && !isIndexing && (
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-2 p-3 bg-yellow-50 rounded-lg border border-yellow-200">
            <button
              type="button"
              onClick={() => handleIndex(true)}
              className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors text-sm font-medium min-h-[44px] sm:min-h-0"
            >
              Force Re-index
            </button>
            <span className="text-sm text-gray-700">Repository already indexed</span>
          </div>
        )}
      </form>

      {isIndexing && (
        <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex items-center gap-3 mb-2">
          <LoadingSpinner size="sm" />
            <span className="text-gray-700 font-medium">{indexingStage || 'Indexing repository...'}</span>
          </div>
          <div className="text-sm text-gray-600 mt-2">
            <p>This may take a few minutes depending on repository size.</p>
            <p className="mt-1">Please don't close this page.</p>
          </div>
        </div>
      )}

      {indexStatus && (
        <div className={`mt-4 p-4 rounded-lg border ${
          indexStatus.status === 'indexed' 
            ? 'bg-green-50 text-green-800 border-green-200' 
            : 'bg-blue-50 text-blue-800 border-blue-200'
        }`}>
          <div className="flex items-start gap-2">
            <svg className="w-5 h-5 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <div>
          <p className="font-medium">
            {indexStatus.status === 'indexed' 
                  ? `Successfully indexed ${indexStatus.chunks_indexed} code chunks`
              : `Repository already indexed (${indexStatus.chunks_indexed} chunks)`}
          </p>
              {indexStatus.status === 'indexed' && (
                <p className="text-sm mt-1 opacity-90">You can now start asking questions about this repository!</p>
              )}
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-50 text-red-800 rounded-lg border border-red-200">
          <div className="flex items-start gap-2">
            <svg className="w-5 h-5 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <div>
              <p className="font-medium">Error</p>
              <p className="text-sm mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {indexedRepos.length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            Indexed Repositories ({indexedRepos.length})
          </h3>
          <div className="space-y-2 max-h-64 overflow-y-auto custom-scrollbar">
            {indexedRepos.map((repo, idx) => (
              <div
                key={idx}
                className={`p-3 rounded-lg border transition-colors ${
                  currentRepo === repo.repo_url
                    ? 'bg-blue-50 border-blue-300'
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <button
                    type="button"
                    onClick={() => handleSelectRepo(repo.repo_url)}
                    className={`flex-1 text-left text-sm font-medium transition-colors ${
                      currentRepo === repo.repo_url
                        ? 'text-blue-700'
                        : 'text-gray-700 hover:text-blue-600'
                    }`}
                    aria-label={`Select repository ${formatRepoName(repo.repo_url)}`}
                  >
                    <div className="flex items-center gap-2">
                      {currentRepo === repo.repo_url && (
                        <svg className="w-4 h-4 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                      )}
                      <span className="font-mono text-xs break-all">{formatRepoName(repo.repo_url)}</span>
                    </div>
                  </button>
                  <button
                    type="button"
                    onClick={() => handleDeleteRepo(repo.repo_url)}
                    className="p-1 text-gray-400 hover:text-red-600 transition-colors rounded"
                    aria-label={`Delete repository ${formatRepoName(repo.repo_url)}`}
                    title="Remove repository"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
