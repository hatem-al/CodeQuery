"""
Authentication and user management module.
"""

import os
# Import PyJWT - check if it's the correct package
try:
    import jwt
    # Verify it's PyJWT by checking for encode method
    if not hasattr(jwt, 'encode'):
        raise ImportError(
            "Wrong 'jwt' package installed. Please uninstall it and install PyJWT:\n"
            "  pip uninstall jwt\n"
            "  pip install PyJWT==2.8.0"
        )
except ImportError as e:
    # Re-raise with the same message if it's our custom error, otherwise provide default message
    if "Wrong 'jwt' package" in str(e):
        raise
    raise ImportError("Please install PyJWT: pip install PyJWT==2.8.0") from e
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict
from pathlib import Path
import json
from database import get_db_connection, init_database, migrate_from_json

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Validate JWT secret key
if SECRET_KEY == "your-secret-key-change-in-production" or len(SECRET_KEY) < 32:
    import warnings
    warnings.warn(
        "⚠️  SECURITY WARNING: JWT_SECRET_KEY is using default or weak value. "
        "Please set a strong JWT_SECRET_KEY environment variable (at least 32 characters) "
        "for production use. Current key length: {} characters".format(len(SECRET_KEY)),
        UserWarning
    )

# User storage path (legacy JSON - kept for migration)
USERS_DB_PATH = Path(__file__).parent.parent / "data" / "users.json"
USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Initialize SQLite database and migrate from JSON if needed
init_database()
# Migrate existing users from JSON to SQLite (only runs once)
try:
    migrated_count = migrate_from_json()
    if migrated_count > 0:
        import warnings
        warnings.warn(
            f"Migrated {migrated_count} users from JSON to SQLite database.",
            UserWarning
        )
except Exception as e:
    import warnings
    warnings.warn(
        f"Error during JSON to SQLite migration: {e}",
        UserWarning
    )


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash using bcrypt."""
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    except Exception:
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict]:
    """Decode and verify a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email from SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    
    if row:
        return dict(row)
    return None


def get_user_by_id(user_id: str) -> Optional[Dict]:
    """Get user by user ID from SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    
    if row:
        return dict(row)
    return None


def create_user(email: str, password: str, username: Optional[str] = None) -> Dict:
    """Create a new user in SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if user already exists
    existing = get_user_by_email(email)
    if existing:
        raise ValueError("User with this email already exists")
    
    # Generate user ID
    # Get current count for numbering
    cursor.execute("SELECT COUNT(*) as count FROM users")
    count = cursor.fetchone()['count']
    user_id = f"user_{count + 1}_{int(datetime.utcnow().timestamp())}"
    
    # Create user
    username_value = username or email.split('@')[0]
    hashed_password = get_password_hash(password)
    created_at = datetime.utcnow().isoformat()
    
    cursor.execute("""
        INSERT INTO users (id, email, username, hashed_password, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, email, username_value, hashed_password, created_at))
    
    conn.commit()
    
    return {
        "id": user_id,
        "email": email,
        "username": username_value,
        "hashed_password": hashed_password,
        "created_at": created_at
    }


def authenticate_user(email: str, password: str) -> Optional[Dict]:
    """Authenticate a user and return user dict if successful."""
    user = get_user_by_email(email)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


