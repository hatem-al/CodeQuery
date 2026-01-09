/**
 * Header component with gradient background and user info
 */
export default function Header({ user, onLogout, darkMode, onToggleDarkMode }) {
  return (
    <header className="bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg flex-shrink-0">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex items-center justify-between">
          {/* Left side - Logo and Tagline */}
          <div className="flex items-center gap-3">
            <div>
              <h1 className="text-2xl font-bold">CodeQuery</h1>
              <p className="text-sm opacity-80">AI-powered code search</p>
            </div>
          </div>

          {/* Right side - User info and actions */}
          <div className="flex items-center gap-4">
            {/* User info */}
            <div className="text-right hidden sm:block">
              <p className="text-sm font-medium">{user.username}</p>
            </div>

            {/* Dark mode toggle */}
            <button
              onClick={onToggleDarkMode}
              className="p-2 rounded-lg bg-white/20 hover:bg-white/30 transition-colors"
              aria-label="Toggle dark mode"
            >
              {darkMode ? (
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                </svg>
              )}
            </button>

            {/* Logout button */}
            <button
              onClick={onLogout}
              className="px-4 py-2 bg-white/20 hover:bg-white/30 rounded-lg transition-colors text-sm font-medium"
              aria-label="Logout"
            >
              Logout
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}

