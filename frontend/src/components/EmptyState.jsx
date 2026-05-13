import { useState } from 'react';
import axios from 'axios';
import LoadingSpinner from './LoadingSpinner';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

const features = [
  {
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    title: 'Instant answers',
    desc: 'Semantic + keyword search across 50K+ lines in milliseconds',
  },
  {
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17H3a2 2 0 01-2-2V5a2 2 0 012-2h14a2 2 0 012 2v10a2 2 0 01-2 2h-2" />
      </svg>
    ),
    title: 'GPT-4o explanations',
    desc: 'Understands logic, not just syntax — explains the why',
  },
  {
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
      </svg>
    ),
    title: 'Multi-language',
    desc: 'Python, JavaScript, TypeScript, Java, C++ and more',
  },
];

export default function EmptyState({ onRepoIndexed }) {
  const [repoUrl, setRepoUrl] = useState('');
  const [isIndexing, setIsIndexing] = useState(false);
  const [error, setError] = useState(null);

  const pollIndexingStatus = async (url) => {
    const maxAttempts = 60;
    let attempts = 0;
    const poll = async () => {
      try {
        const encoded = encodeURIComponent(url);
        const { data } = await axios.get(`${API_BASE_URL}/index/status/${encoded}`);
        if (data.status === 'completed' || data.status === 'already_indexed') {
          onRepoIndexed?.(url);
          setRepoUrl('');
          setIsIndexing(false);
        } else if (data.status === 'error') {
          setError(data.error || 'Failed to index. Please try again.');
          setIsIndexing(false);
        } else {
          attempts++;
          if (attempts < maxAttempts) setTimeout(poll, 5000);
          else { setError('Indexing timed out.'); setIsIndexing(false); }
        }
      } catch (err) {
        attempts++;
        if (err.response?.status === 404 && attempts < maxAttempts) setTimeout(poll, 5000);
        else { setError('Error checking status.'); setIsIndexing(false); }
      }
    };
    setTimeout(poll, 2000);
  };

  const handleIndex = async () => {
    const trimmed = repoUrl.trim();
    if (!trimmed) { setError('Please enter a GitHub repository URL'); return; }
    const valid = /^https?:\/\/(www\.)?github\.com\/[\w\-\.]+\/[\w\-\.]+(\.git)?(\/.*)?$/.test(trimmed);
    if (!valid) { setError('Please enter a valid GitHub repository URL'); return; }

    setIsIndexing(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const { data } = await axios.post(
        `${API_BASE_URL}/index`,
        { repo_url: trimmed, force_reindex: false },
        { headers: { Authorization: `Bearer ${token}` }, timeout: 60000 }
      );
      if (data.status === 'indexed' || data.status === 'already_indexed') {
        onRepoIndexed?.(data.repo_id);
        setIsIndexing(false);
      } else {
        pollIndexingStatus(trimmed);
      }
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        (err.code === 'ERR_NETWORK' ? 'Cannot connect to backend.' : 'Failed to index. Please try again.')
      );
      setIsIndexing(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] px-4 py-16">
      {/* Icon */}
      <div className="w-16 h-16 bg-indigo-50 dark:bg-indigo-950/50 border border-indigo-100 dark:border-indigo-900 rounded-2xl flex items-center justify-center mb-6 text-indigo-600 dark:text-indigo-400">
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
        </svg>
      </div>

      <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-3 text-center">
        Ask your codebase anything
      </h1>
      <p className="text-gray-500 dark:text-slate-400 text-base sm:text-lg mb-10 text-center max-w-xl">
        Paste a public GitHub URL to index it, then ask questions in plain English.
      </p>

      {/* Input */}
      <div className="w-full max-w-2xl mb-12">
        <div className="flex gap-2 p-1.5 bg-white dark:bg-slate-900 border border-gray-200 dark:border-slate-700 rounded-xl shadow-sm">
          <div className="flex items-center pl-3 text-gray-400 dark:text-slate-500 flex-shrink-0">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
          </div>
          <input
            type="text"
            value={repoUrl}
            onChange={(e) => { setRepoUrl(e.target.value); setError(null); }}
            onKeyDown={(e) => e.key === 'Enter' && !isIndexing && handleIndex()}
            placeholder="https://github.com/username/repo"
            className="flex-1 px-2 py-2.5 bg-transparent text-sm text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-slate-500 focus:outline-none"
            disabled={isIndexing}
          />
          <button
            onClick={handleIndex}
            disabled={isIndexing || !repoUrl.trim()}
            className="px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-700 hover:to-violet-700 text-white rounded-lg font-semibold text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 flex-shrink-0"
          >
            {isIndexing ? (
              <><LoadingSpinner size="sm" /><span>Indexing…</span></>
            ) : 'Index Repository'}
          </button>
        </div>

        {error && (
          <p className="mt-2 text-sm text-red-600 dark:text-red-400 flex items-center gap-1.5 px-1">
            <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            {error}
          </p>
        )}

        {isIndexing && (
          <div className="mt-3 flex items-center gap-2 px-1 text-sm text-indigo-600 dark:text-indigo-400">
            <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
            Indexing… this takes a few minutes for large repos.
          </div>
        )}
      </div>

      {/* Feature cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 w-full max-w-3xl">
        {features.map((f) => (
          <div
            key={f.title}
            className="bg-white dark:bg-slate-900 border border-gray-200 dark:border-slate-800 rounded-xl p-5 hover:border-indigo-200 dark:hover:border-indigo-900 hover:shadow-md transition-all"
          >
            <div className="w-9 h-9 bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 rounded-lg flex items-center justify-center mb-3">
              {f.icon}
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white text-sm mb-1">{f.title}</h3>
            <p className="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">{f.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
