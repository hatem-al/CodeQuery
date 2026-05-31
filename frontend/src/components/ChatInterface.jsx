import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import LoadingSpinner from './LoadingSpinner';
import CodeBlock from './CodeBlock';
import SourcesList from './SourcesList';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

// Helper to get auth token
const getAuthToken = () => {
  return localStorage.getItem('token');
};

// Register the axios auth interceptor once at module load time
let _interceptorRegistered = false;
if (!_interceptorRegistered) {
  _interceptorRegistered = true;
  axios.interceptors.request.use((config) => {
    const token = getAuthToken();
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  });
}

/**
 * ChatInterface component - Chat messages, input box, send button
 */
export default function ChatInterface({ repoId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const messagesContainerRef = useRef(null);

  // Get user ID for chat history isolation
  const getUserId = () => {
    try {
      const user = localStorage.getItem('user');
      return user ? JSON.parse(user).id : 'anonymous';
    } catch {
      return 'anonymous';
    }
  };

  // Load chat history from localStorage (per user and repo)
  useEffect(() => {
    if (repoId) {
      const userId = getUserId();
      const storageKey = `chat_history_${userId}_${repoId}`;
      try {
        const savedMessages = localStorage.getItem(storageKey);
        if (savedMessages) {
          const parsed = JSON.parse(savedMessages);
          // Convert timestamp strings back to Date objects
          const messagesWithDates = parsed.map(msg => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }));
          setMessages(messagesWithDates);
        }
      } catch (err) {
        console.error('Error loading chat history:', err);
      }
    }
  }, [repoId]);

  // Save chat history to localStorage (per user and repo)
  useEffect(() => {
    if (repoId && messages.length > 0) {
      const userId = getUserId();
      const storageKey = `chat_history_${userId}_${repoId}`;
      try {
        // Exclude in-progress streaming messages; cap at 50 to avoid storage overflow
        const toSave = messages
          .filter(m => !m.streaming)
          .slice(-50);
        localStorage.setItem(storageKey, JSON.stringify(toSave));
      } catch (err) {
        console.error('Error saving chat history:', err);
      }
    }
  }, [messages, repoId]);

  const isAtBottomRef = useRef(true);

  // Track whether the user is near the bottom
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;
    const onScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      isAtBottomRef.current = scrollHeight - scrollTop - clientHeight < 150;
    };
    container.addEventListener('scroll', onScroll, { passive: true });
    return () => container.removeEventListener('scroll', onScroll);
  }, []);

  const scrollToBottom = (smooth = true) => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto' });
    }
  };

  useEffect(() => {
    const last = messages[messages.length - 1];
    if (!last) return;
    // Always scroll for user messages or when streaming finishes;
    // during streaming only scroll if already at bottom
    if (last.role === 'user' || !last.streaming) {
      scrollToBottom();
    } else if (isAtBottomRef.current) {
      scrollToBottom(false);
    }
  }, [messages]);

  useEffect(() => {
    // Focus input when repo changes
    inputRef.current?.focus();
    // Clear messages when repo changes
    if (repoId) {
      setMessages([]);
      setError(null);
    }
  }, [repoId]);

  const getErrorMessage = (err) => {
    if (!err) return 'Failed to get response. Please try again.';
    const msg = err.message || '';
    if (msg.includes('Failed to fetch') || msg.includes('NetworkError') || msg.includes('Network Error'))
      return 'Cannot connect to backend. Please make sure the server is running.';
    if (msg.includes('404') || msg.includes('not found'))
      return 'Repository not found. Please re-index the repository.';
    if (msg.includes('429') || msg.includes('rate limit'))
      return 'Rate limit exceeded. Please wait a moment and try again.';
    return msg || 'Failed to get response. Please try again.';
  };

  const handleSend = async (e) => {
    e.preventDefault();

    if (!input.trim() || isLoading) return;
    if (!repoId) { setError('Please index a repository first'); return; }

    const userMessage = input.trim();
    setInput('');
    setError(null);

    // Build history from settled messages only (no error/streaming messages)
    const chatHistory = messages
      .filter(m => (m.role === 'user' || m.role === 'assistant') && !m.streaming)
      .slice(-10)
      .map(m => ({ role: m.role, content: m.content }));

    setMessages(prev => [...prev, { role: 'user', content: userMessage, timestamp: new Date() }]);
    setIsLoading(true);

    try {
      const token = getAuthToken();
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ query: userMessage, repo_id: repoId, chat_history: chatHistory })
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let assistantAdded = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        // SSE events are separated by double newlines
        const events = buffer.split('\n\n');
        buffer = events.pop() ?? '';

        for (const event of events) {
          if (!event.startsWith('data: ')) continue;
          let data;
          try { data = JSON.parse(event.slice(6).trim()); } catch { continue; }

          if (data.error) throw new Error(data.error);

          // Single-event answer (e.g. "no results found" case)
          if (data.answer !== undefined) {
            setIsLoading(false);
            assistantAdded = true;
            setMessages(prev => [...prev, {
              role: 'assistant',
              content: data.answer,
              sources: data.sources || [],
              timestamp: new Date(),
              streaming: false
            }]);
          }

          // Sources arrive first — add the assistant message placeholder
          if (data.sources && !assistantAdded) {
            assistantAdded = true;
            setIsLoading(false);
            setMessages(prev => [...prev, {
              role: 'assistant',
              content: '',
              sources: data.sources,
              timestamp: new Date(),
              streaming: true
            }]);
          }

          // Append streamed token to last assistant message
          if (data.content) {
            setMessages(prev => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last?.role === 'assistant') {
                updated[updated.length - 1] = { ...last, content: last.content + data.content };
              }
              return updated;
            });
          }

          // Clear sources if LLM fired the scope guard
          if (data.clear_sources) {
            setMessages(prev => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last?.role === 'assistant') {
                updated[updated.length - 1] = { ...last, sources: [] };
              }
              return updated;
            });
          }

          // Mark stream complete
          if (data.done) {
            setMessages(prev => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last?.role === 'assistant') {
                updated[updated.length - 1] = { ...last, streaming: false };
              }
              return updated;
            });
          }
        }
      }

      setIsLoading(false);
    } catch (err) {
      setIsLoading(false);
      const errorContent = getErrorMessage(err);
      // Drop any empty streaming placeholder before showing the error
      setMessages(prev => {
        const cleaned = prev.filter(m => !(m.streaming && m.content === ''));
        return [...cleaned, { role: 'error', content: errorContent, timestamp: new Date() }];
      });
      setError(errorContent);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend(e);
    }
  };

  const clearChat = () => {
    if (messages.length > 0 && window.confirm('Are you sure you want to clear the chat history?')) {
    setMessages([]);
    setError(null);
      // Clear from localStorage (per user and repo)
      if (repoId) {
        const userId = getUserId();
        const storageKey = `chat_history_${userId}_${repoId}`;
        localStorage.removeItem(storageKey);
      }
    }
  };

  const handleExampleClick = (example) => {
    setInput(example);
    inputRef.current?.focus();
  };

  // Extract code blocks from message content (handles inline code and code blocks)
  const extractCodeBlocks = (content) => {
    const parts = [];
    let lastIndex = 0;
    
    // Match code blocks: ```language\ncode```
    const codeBlockRegex = /```(\w+)?\n?([\s\S]*?)```/g;
    let match;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      // Add text before code block
      if (match.index > lastIndex) {
        const textBefore = content.substring(lastIndex, match.index);
        // Also extract inline code from text before
        parts.push(...extractInlineCode(textBefore));
      }

      // Add code block
      parts.push({
        type: 'code',
        language: match[1] || 'text',
        content: match[2].trim()
      });

      lastIndex = match.index + match[0].length;
    }

    // Add remaining text
    if (lastIndex < content.length) {
      const remainingText = content.substring(lastIndex);
      parts.push(...extractInlineCode(remainingText));
    }

    return parts.length > 0 ? parts : [{ type: 'text', content }];
  };

  // Extract inline code: `code`
  const extractInlineCode = (text) => {
    const parts = [];
    let lastIndex = 0;
    const inlineCodeRegex = /`([^`]+)`/g;
    let match;

    while ((match = inlineCodeRegex.exec(text)) !== null) {
      // Add text before inline code
      if (match.index > lastIndex) {
        parts.push({
          type: 'text',
          content: text.substring(lastIndex, match.index)
        });
      }

      // Add inline code
      parts.push({
        type: 'inline-code',
        content: match[1]
      });

      lastIndex = match.index + match[0].length;
    }

    // Add remaining text
    if (lastIndex < text.length) {
      parts.push({
        type: 'text',
        content: text.substring(lastIndex)
      });
    }

    return parts.length > 0 ? parts : [{ type: 'text', content: text }];
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const exampleQueries = [
    "How does authentication work?",
    "Where is the database configured?",
    "Show me the main UI components",
    "What API endpoints are available?"
  ];

  const userInitial = (() => {
    try {
      const u = localStorage.getItem('user');
      const parsed = u ? JSON.parse(u) : null;
      return (parsed?.username || parsed?.email || '?')[0].toUpperCase();
    } catch { return '?'; }
  })();

  return (
    <div className="flex flex-col h-full bg-white dark:bg-slate-900 rounded-xl border border-gray-200 dark:border-slate-800 shadow-sm transition-all">
      {/* Chat Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-slate-800">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 bg-gradient-to-br from-indigo-500 to-violet-600 rounded-lg flex items-center justify-center">
            <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
          </div>
          <h2 className="font-bold text-gray-900 dark:text-white text-sm">Chat with Codebase</h2>
        </div>
        {messages.length > 0 && (
          <button
            onClick={clearChat}
            className="text-xs text-gray-400 dark:text-slate-500 hover:text-gray-600 dark:hover:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-800 px-2.5 py-1.5 rounded-lg transition-colors"
            aria-label="Clear chat history"
          >
            Clear
          </button>
        )}
      </div>

      {/* Messages Area */}
      <div
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto p-4 space-y-5 custom-scrollbar bg-slate-50/50 dark:bg-slate-950/30"
        style={{ maxHeight: 'calc(100vh - 12rem)' }}
      >
        {/* Empty State */}
        {messages.length === 0 && !isLoading && repoId && (
          <div className="flex flex-col items-center justify-center h-full px-4 pt-8">
            <div className="w-12 h-12 bg-indigo-50 dark:bg-indigo-950/50 border border-indigo-100 dark:border-indigo-900 rounded-xl flex items-center justify-center mb-5 text-indigo-600 dark:text-indigo-400">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
            </div>
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-1">Ask anything about this repo</h3>
            <p className="text-sm text-gray-500 dark:text-slate-400 mb-7">Try one of these to get started</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-xl">
              {exampleQueries.map((query, index) => (
                <button
                  key={index}
                  onClick={() => handleExampleClick(query)}
                  className="px-4 py-3 text-left bg-white dark:bg-slate-900 hover:bg-indigo-50 dark:hover:bg-indigo-950/40 border border-gray-200 dark:border-slate-800 hover:border-indigo-200 dark:hover:border-indigo-900 rounded-xl transition-all text-sm text-gray-700 dark:text-slate-300"
                >
                  {query}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex gap-2.5 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {/* Bot avatar */}
            {message.role === 'assistant' && (
              <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center flex-shrink-0 mt-1 shadow-sm">
                <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17H3a2 2 0 01-2-2V5a2 2 0 012-2h14a2 2 0 012 2v10a2 2 0 01-2 2h-2" />
                </svg>
              </div>
            )}

            <div className={message.role === 'user' ? 'max-w-[72%]' : 'max-w-[82%] min-w-0'}>
              <div
                className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  message.role === 'user'
                    ? 'bg-gradient-to-br from-indigo-600 to-violet-600 text-white rounded-tr-sm shadow-sm shadow-indigo-500/20'
                    : message.role === 'error'
                    ? 'bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-900 rounded-tl-sm'
                    : 'bg-white dark:bg-slate-900 text-gray-900 dark:text-slate-100 border border-gray-200 dark:border-slate-800 rounded-tl-sm shadow-sm'
                }`}
              >
                {message.role === 'user' ? (
                  <p className="whitespace-pre-wrap break-words" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>{message.content}</p>
                ) : (
                  <div>
                    {(() => {
                      const parts = extractCodeBlocks(message.content);
                      const rendered = [];
                      let i = 0;
                      while (i < parts.length) {
                        if (parts[i].type === 'code') {
                          rendered.push(
                            <div key={i} className="my-3 -mx-1">
                              <CodeBlock code={parts[i].content} language={parts[i].language} />
                            </div>
                          );
                          i++;
                        } else {
                          const groupStart = i;
                          const group = [];
                          while (i < parts.length && parts[i].type !== 'code') {
                            group.push(parts[i]);
                            i++;
                          }
                          rendered.push(
                            <p key={groupStart} className="whitespace-pre-wrap mb-2 last:mb-0 break-words" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>
                              {group.map((p, j) =>
                                p.type === 'inline-code' ? (
                                  <code key={j} className="bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded text-[13px] font-mono text-indigo-700 dark:text-indigo-300 border border-slate-200 dark:border-slate-700">
                                    {p.content}
                                  </code>
                                ) : p.content
                              )}
                            </p>
                          );
                        }
                      }
                      return rendered;
                    })()}

                    {message.streaming && (
                      <span className="inline-block w-1.5 h-3.5 bg-indigo-400 rounded-sm animate-pulse ml-0.5 align-middle" />
                    )}

                    {message.sources && message.sources.length > 0 && (
                      <SourcesList sources={message.sources} repoUrl={repoId} />
                    )}
                  </div>
                )}
              </div>
              <span className="text-[11px] text-gray-400 dark:text-slate-600 mt-1 block px-1">
                {formatTimestamp(message.timestamp)}
              </span>
            </div>

            {/* User avatar */}
            {message.role === 'user' && (
              <div className="w-7 h-7 rounded-lg bg-slate-200 dark:bg-slate-700 flex items-center justify-center flex-shrink-0 mt-1 text-xs font-bold text-slate-600 dark:text-slate-300">
                {userInitial}
              </div>
            )}
          </div>
        ))}

        {/* Loading State */}
        {isLoading && (
          <div className="flex gap-2.5 justify-start">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center flex-shrink-0 mt-1">
              <svg className="w-3.5 h-3.5 text-white animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17H3a2 2 0 01-2-2V5a2 2 0 012-2h14a2 2 0 012 2v10a2 2 0 01-2 2h-2" />
              </svg>
            </div>
            <div className="bg-white dark:bg-slate-900 border border-gray-200 dark:border-slate-800 rounded-2xl rounded-tl-sm px-4 py-3 max-w-[80%] shadow-sm">
              <div className="flex items-center gap-1.5 mb-3">
                <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                <span className="text-xs text-gray-500 dark:text-slate-400 ml-1">Thinking…</span>
              </div>
              <div className="space-y-2">
                <div className="h-2.5 bg-slate-100 dark:bg-slate-800 rounded-full animate-pulse w-full" />
                <div className="h-2.5 bg-slate-100 dark:bg-slate-800 rounded-full animate-pulse w-5/6" />
                <div className="h-2.5 bg-slate-100 dark:bg-slate-800 rounded-full animate-pulse w-3/5" />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 dark:border-slate-800 p-3 bg-white dark:bg-slate-900 rounded-b-xl">
        {error && (
          <div className="mb-2 flex items-start gap-2 p-2.5 bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-400 rounded-lg text-xs border border-red-200 dark:border-red-900">
            <svg className="w-4 h-4 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <span>{error}</span>
          </div>
        )}

        {!repoId && (
          <div className="mb-2 flex items-center gap-2 p-2.5 bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-400 rounded-lg text-xs border border-amber-200 dark:border-amber-900">
            <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            Select or index a repository first
          </div>
        )}

        <div className="flex gap-2 p-1.5 bg-slate-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl focus-within:border-indigo-400 dark:focus-within:border-indigo-700 transition-colors">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => { setInput(e.target.value); setError(null); }}
            onKeyDown={handleKeyPress}
            placeholder={repoId ? 'Ask about your codebase…' : 'Select a repository first…'}
            disabled={!repoId || isLoading}
            maxLength={500}
            rows={1}
            className="flex-1 px-3 py-2 bg-transparent text-sm text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-slate-500 resize-none focus:outline-none disabled:cursor-not-allowed"
            aria-label="Chat input"
            style={{ maxHeight: '120px' }}
          />
          <button
            type="button"
            onClick={handleSend}
            disabled={!repoId || !input.trim() || isLoading}
            className="px-4 py-2 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-700 hover:to-violet-700 text-white rounded-lg font-medium text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5 flex-shrink-0 shadow-sm shadow-indigo-500/20"
            aria-label="Send message"
          >
            {isLoading ? (
              <LoadingSpinner size="sm" />
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            )}
            <span className="hidden sm:inline">{isLoading ? 'Sending…' : 'Send'}</span>
          </button>
        </div>
        <p className="text-[11px] text-gray-400 dark:text-slate-600 mt-1.5 px-1">
          <kbd className="px-1 py-0.5 bg-gray-100 dark:bg-slate-800 rounded text-[11px] border border-gray-200 dark:border-slate-700">Enter</kbd> to send &middot; <kbd className="px-1 py-0.5 bg-gray-100 dark:bg-slate-800 rounded text-[11px] border border-gray-200 dark:border-slate-700">Shift+Enter</kbd> for new line
        </p>
      </div>
    </div>
  );
}
