"""
Database module for SQLite user storage.
"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict
from contextlib import contextmanager
import threading

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "users.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Thread-local storage for connections (SQLite is not thread-safe by default)
_local = threading.local()


def get_db_connection():
    """Get a thread-local database connection."""
    if not hasattr(_local, 'connection') or _local.connection is None:
        _local.connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.connection.row_factory = sqlite3.Row  # Return rows as dict-like objects
    return _local.connection


def init_database():
    """Initialize the database with the users table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            username TEXT NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    # Create index on email for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
    """)
    
    conn.commit()


def migrate_from_json():
    """Migrate users from JSON file to SQLite database."""
    json_path = Path(__file__).parent.parent / "data" / "users.json"
    
    if not json_path.exists():
        return 0
    
    import json
    
    try:
        with open(json_path, 'r') as f:
            users_data = json.load(f)
        
        if not users_data:
            return 0
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        migrated = 0
        for email, user_data in users_data.items():
            try:
                # Check if user already exists
                cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
                if cursor.fetchone():
                    continue  # User already exists, skip
                
                cursor.execute("""
                    INSERT INTO users (id, email, username, hashed_password, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    user_data.get('id'),
                    user_data.get('email'),
                    user_data.get('username'),
                    user_data.get('hashed_password'),
                    user_data.get('created_at')
                ))
                migrated += 1
            except sqlite3.IntegrityError:
                # User already exists, skip
                pass
        
        conn.commit()
        return migrated
    except Exception as e:
        print(f"Error migrating users from JSON: {e}")
        return 0


# Initialize database on import
init_database()

