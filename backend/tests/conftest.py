"""
Pytest configuration: redirect SQLite to a temp file and set env vars
before any module under test is imported.
"""

import os
import tempfile
from pathlib import Path

import pytest

# ── env vars ──────────────────────────────────────────────────────────────────
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-tests-only-32chars!!")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key-not-real")

# ── redirect DB to a temp file (must happen before auth/database is imported) ─
import database  # noqa: E402  (import after env setup intentional)

_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_TMP_DB_PATH = Path(_tmp_db.name)
_tmp_db.close()

database.DB_PATH = _TMP_DB_PATH
# Clear any cached connection that pointed at the real DB
if getattr(database._local, "connection", None) is not None:
    try:
        database._local.connection.close()
    except Exception:
        pass
    database._local.connection = None
database.init_database()


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_users_table():
    """Wipe the users table between tests to keep them independent."""
    conn = database.get_db_connection()
    conn.execute("DELETE FROM users")
    conn.commit()
    yield
    conn.execute("DELETE FROM users")
    conn.commit()


