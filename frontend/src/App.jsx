import { useState, useEffect } from 'react';
import axios from 'axios';
import Login from './components/Login';
import Header from './components/Header';
import EmptyState from './components/EmptyState';
import RepoInput from './components/RepoInput';
import ChatInterface from './components/ChatInterface';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

function App() {
  const [currentRepo, setCurrentRepo] = useState(null);
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [darkMode, setDarkMode] = useState(() => {
    // Check localStorage for saved theme preference
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });

  // Check for existing auth on mount
  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    const savedUser = localStorage.getItem('user');
    
    if (savedToken && savedUser) {
      setToken(savedToken);
      setUser(JSON.parse(savedUser));
      // Verify token is still valid
      verifyToken(savedToken);
    }
  }, []);

  const verifyToken = async (tokenToVerify) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/auth/me`, {
        headers: { Authorization: `Bearer ${tokenToVerify}` }
      });
      setUser(response.data);
    } catch (err) {
      // Token invalid, clear auth
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      setToken(null);
      setUser(null);
    }
  };

  const handleLogin = (userData, authToken) => {
    setUser(userData);
    setToken(authToken);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setToken(null);
    setUser(null);
    setCurrentRepo(null);
  };

  // Apply dark mode class to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
  }, [darkMode]);

  const [indexedRepos, setIndexedRepos] = useState([]);
  const [loadingRepos, setLoadingRepos] = useState(true);

  // Load indexed repos
  useEffect(() => {
    if (user && token) {
      loadIndexedRepos();
    }
  }, [user, token]);

  const loadIndexedRepos = async () => {
    try {
      setLoadingRepos(true);
      const response = await axios.get(`${API_BASE_URL}/repos`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setIndexedRepos(response.data.repos || []);
    } catch (err) {
      console.error('Error loading repos:', err);
    } finally {
      setLoadingRepos(false);
    }
  };

  const handleRepoIndexed = (repoId) => {
    setCurrentRepo(repoId);
    loadIndexedRepos(); // Reload repos list
  };

  // Show login if not authenticated
  if (!user || !token) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 flex flex-col transition-colors">
      {/* Header */}
      <Header 
        user={user}
        onLogout={handleLogout}
        darkMode={darkMode}
        onToggleDarkMode={() => setDarkMode(!darkMode)}
      />

      {/* Main Content */}
      <main className="flex-1 w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {!loadingRepos && indexedRepos.length === 0 ? (
          /* Empty State - No repos indexed */
          <EmptyState onRepoIndexed={handleRepoIndexed} />
        ) : (
          /* Main App - With repos */
          <div className="max-w-7xl mx-auto">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
              {/* Left Column - Repository Input */}
              <div className="lg:col-span-1 flex flex-col">
                <RepoInput 
                  onRepoIndexed={handleRepoIndexed}
                  currentRepo={currentRepo}
                />
              </div>

              {/* Right Column - Chat Interface */}
              <div className="lg:col-span-2 flex flex-col min-h-0">
                <div className="flex-1 min-h-[500px] sm:min-h-[600px]">
                  <ChatInterface repoId={currentRepo} />
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 flex-shrink-0 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 sm:py-4">
          <p className="text-center text-xs sm:text-sm text-gray-600 dark:text-gray-400">
            Made by Hatem Almasri
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
