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

// Configure axios to include token in requests (if not already configured)
if (!axios.defaults.headers.common['Authorization']) {
  axios.interceptors.request.use((config) => {
    const token = getAuthToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
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
        localStorage.setItem(storageKey, JSON.stringify(messages));
      } catch (err) {
        console.error('Error saving chat history:', err);
      }
    }
  }, [messages, repoId]);

  const scrollToBottom = (smooth = true) => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto' });
    }
  };

  useEffect(() => {
    scrollToBottom();
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

  const handleSend = async (e) => {
    e.preventDefault();
    
    if (!input.trim() || isLoading) return;
    
    if (!repoId) {
      setError('Please index a repository first');
      return;
    }

    const userMessage = input.trim();
    setInput('');
    setError(null);

    // Add user message immediately
    const newUserMessage = {
      role: 'user',
      content: userMessage,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, newUserMessage]);

    setIsLoading(true);

    try {
      // Call chat endpoint with timeout
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        query: userMessage,
        repo_id: repoId
      }, {
        timeout: 60000 // 60 second timeout
      });

      const assistantMessage = {
        role: 'assistant',
        content: response.data.answer || 'No response received.',
        sources: response.data.sources || [],
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      let errorContent = 'Failed to get response. Please try again.';
      
      if (err.code === 'ERR_NETWORK' || err.message === 'Network Error') {
        errorContent = 'Cannot connect to backend. Please make sure the server is running.';
      } else if (err.response?.status === 404) {
        errorContent = 'Repository not found. Please re-index the repository.';
      } else if (err.response?.status === 429) {
        errorContent = 'Rate limit exceeded. Please wait a moment and try again.';
      } else if (err.response?.data?.detail) {
        errorContent = err.response.data.detail;
      } else if (err.message) {
        errorContent = err.message;
      }
      
      const errorMessage = {
        role: 'error',
        content: errorContent,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      setError(errorContent);
    } finally {
      setIsLoading(false);
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

  return (
    <div className="flex flex-col h-full bg-white dark:bg-gray-800 rounded-lg shadow-md transition-all">
      {/* Chat Header */}
      <div className="flex items-center justify-between p-3 sm:p-4 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-lg sm:text-xl font-bold text-gray-800 dark:text-white">Chat with Codebase</h2>
        {messages.length > 0 && (
          <button
            onClick={clearChat}
            className="px-3 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors min-h-[44px] sm:min-h-0"
            aria-label="Clear chat history"
          >
            <span className="hidden sm:inline">Clear Chat</span>
            <span className="sm:hidden">Clear</span>
          </button>
        )}
      </div>

      {/* Messages Area */}
      <div 
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto p-3 sm:p-4 space-y-4 custom-scrollbar"
        style={{ maxHeight: 'calc(100vh - 12rem)' }}
      >
        {/* Empty State */}
        {messages.length === 0 && !isLoading && repoId && (
          <div className="flex flex-col items-center justify-center h-full px-4 pt-8">
            <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center mb-6 transition-all">
              <svg className="w-8 h-8 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-gray-800 dark:text-white mb-2">Start exploring your codebase</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-8">Try one of these example queries to get started</p>
            
            <div className="grid grid-cols-2 gap-3 w-full max-w-2xl">
              {exampleQueries.map((query, index) => (
                <button
                  key={index}
                  onClick={() => handleExampleClick(query)}
                  className="px-4 py-3 text-left bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 border border-gray-200 dark:border-gray-600 rounded-lg transition-all hover:shadow-md hover:border-blue-300 dark:hover:border-blue-600 text-sm text-gray-700 dark:text-gray-300"
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
            className={`flex mb-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'} transition-all`}
          >
            <div className={message.role === 'user' ? 'max-w-[70%]' : 'max-w-[80%]'}>
              <div
                className={`rounded-2xl px-4 py-3 transition-all ${
                  message.role === 'user'
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-tr-sm'
                    : message.role === 'error'
                    ? 'bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-300 border border-red-200 dark:border-red-800 rounded-tl-sm'
                    : 'bg-gray-100 text-gray-900 dark:bg-gray-700 dark:text-gray-200 rounded-tl-sm'
                }`}
              >
                {message.role === 'user' ? (
                  <p className="whitespace-pre-wrap break-words" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>{message.content}</p>
                ) : (
                  <div>
                    {extractCodeBlocks(message.content).map((part, partIndex) => {
                      if (part.type === 'code') {
                        return (
                          <div key={partIndex} className="my-2 -mx-1 sm:-mx-2">
                            <CodeBlock
                              code={part.content}
                              language={part.language}
                            />
                          </div>
                        );
                      } else if (part.type === 'inline-code') {
                        return (
                          <code
                            key={partIndex}
                            className="bg-gray-200 dark:bg-gray-700 px-1.5 py-0.5 rounded text-sm font-mono text-gray-800 dark:text-gray-200"
                          >
                            {part.content}
                          </code>
                        );
                      } else {
                        return (
                          <p key={partIndex} className="whitespace-pre-wrap mb-2 break-words" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>
                            {part.content}
                          </p>
                        );
                      }
                    })}
                    
                    {message.sources && message.sources.length > 0 && (
                      <SourcesList sources={message.sources} repoUrl={repoId} />
                    )}
                  </div>
                )}
              </div>
              <span className="text-xs opacity-75 mt-1 block px-2">
                {formatTimestamp(message.timestamp)}
              </span>
            </div>
          </div>
        ))}

        {/* Loading State */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 dark:bg-gray-700 rounded-2xl rounded-tl-sm p-4 max-w-[80%]">
              <div className="flex items-center gap-3 mb-3">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
                <span className="text-sm text-gray-600 dark:text-gray-300">AI is thinking...</span>
              </div>
              {/* Skeleton loading bars */}
              <div className="space-y-2">
                <div className="h-3 bg-gray-200 dark:bg-gray-600 rounded animate-pulse w-full"></div>
                <div className="h-3 bg-gray-200 dark:bg-gray-600 rounded animate-pulse w-5/6"></div>
                <div className="h-3 bg-gray-200 dark:bg-gray-600 rounded animate-pulse w-4/6"></div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 dark:border-gray-700 p-3 sm:p-4">
        {error && (
          <div className="mb-2 p-2 bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-300 rounded-lg text-sm border border-red-200 dark:border-red-800 transition-all">
            <div className="flex items-start gap-2">
              <svg className="w-4 h-4 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <span>{error}</span>
            </div>
          </div>
        )}
        
        {!repoId && (
          <div className="mb-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-300 rounded-lg text-sm border border-yellow-200 dark:border-yellow-800 transition-all">
            <div className="flex items-start gap-2">
              <svg className="w-4 h-4 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <span>Please index a repository first to start chatting</span>
            </div>
          </div>
        )}

        <form onSubmit={handleSend} className="flex flex-col sm:flex-row gap-2">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                setError(null);
              }}
              onKeyDown={handleKeyPress}
              placeholder={repoId ? "Ask about your codebase... (Shift+Enter for new line)" : "Index a repository first..."}
              disabled={!repoId || isLoading}
              maxLength={500}
              className="w-full px-4 py-3 pr-16 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:cursor-not-allowed transition-all min-h-[44px] bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
              rows={2}
              aria-label="Chat input"
            />
            <div className="absolute bottom-2 right-2 text-xs text-gray-400 dark:text-gray-500">
              {input.length}/500
            </div>
          </div>
          <button
            type="submit"
            disabled={!repoId || !input.trim() || isLoading}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 font-medium min-h-[44px] sm:min-h-0 shadow-sm hover:shadow-md"
            aria-label="Send message"
          >
            {isLoading ? (
              <>
                <LoadingSpinner size="sm" />
                <span className="hidden sm:inline">Sending...</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
                <span className="hidden sm:inline">Send</span>
              </>
            )}
          </button>
        </form>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 transition-all">
          Press <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-xs">Enter</kbd> to send, <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-xs">Shift+Enter</kbd> for new line
        </p>
      </div>
    </div>
  );
}
