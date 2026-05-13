"""
Integration tests for the FastAPI endpoints.

Uses httpx.AsyncClient with ASGITransport to test the app in-process without
a live server. External calls (ChromaDB, OpenAI) are mocked.
"""

import httpx
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch, AsyncMock


# ── app fixture ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_chroma = MagicMock()
    mock_chroma.get_or_create_collection.return_value = mock_collection

    with patch("embeddings.get_chromadb_client", return_value=mock_chroma):
        from main import app as _app
        return _app


@pytest.fixture(autouse=True)
def reset_rate_limiters(app):
    """Clear slowapi's in-memory counters before every test."""
    import main as _main
    try:
        _main.limiter._storage.reset()
        _main.user_limiter._storage.reset()
    except Exception:
        pass


@pytest_asyncio.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── helpers ──────────────────────────────────────────────────────────────────

async def _register(client, email="user@test.com", password="password1"):
    return await client.post("/auth/register", json={"email": email, "password": password})


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── infrastructure endpoints ─────────────────────────────────────────────────

@pytest.mark.asyncio
class TestInfraEndpoints:
    async def test_ping(self, client):
        r = await client.get("/ping")
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

    async def test_health(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        assert "status" in r.json()

    async def test_root(self, client):
        r = await client.get("/")
        assert r.status_code == 200
        assert "endpoints" in r.json()


# ── auth: register ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRegister:
    async def test_valid_registration_returns_token(self, client):
        r = await _register(client, "new@test.com")
        assert r.status_code == 200
        data = r.json()
        assert "access_token" in data
        assert data["user"]["email"] == "new@test.com"

    async def test_duplicate_email_rejected(self, client):
        await _register(client, "dup@test.com")
        r = await _register(client, "dup@test.com")
        assert r.status_code == 400

    async def test_password_too_short(self, client):
        r = await _register(client, "short@test.com", password="abc1")
        assert r.status_code == 400

    async def test_password_letters_only(self, client):
        r = await _register(client, "nonum@test.com", password="onlyletters")
        assert r.status_code == 400

    async def test_password_digits_only(self, client):
        r = await _register(client, "nolet@test.com", password="12345678")
        assert r.status_code == 400

    async def test_invalid_email_rejected(self, client):
        r = await client.post("/auth/register",
                              json={"email": "notanemail", "password": "password1"})
        assert r.status_code == 400


# ── auth: login ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestLogin:
    async def test_valid_login_returns_token(self, client):
        await _register(client, "login@test.com")
        r = await client.post("/auth/login",
                              json={"email": "login@test.com", "password": "password1"})
        assert r.status_code == 200
        assert "access_token" in r.json()

    async def test_wrong_password_rejected(self, client):
        await _register(client, "login2@test.com")
        r = await client.post("/auth/login",
                              json={"email": "login2@test.com", "password": "badpass"})
        assert r.status_code == 401

    async def test_unknown_user_rejected(self, client):
        r = await client.post("/auth/login",
                              json={"email": "ghost@test.com", "password": "password1"})
        assert r.status_code == 401


# ── auth: /me ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestMe:
    async def test_me_without_token(self, client):
        r = await client.get("/auth/me")
        assert r.status_code == 403

    async def test_me_with_valid_token(self, client):
        reg = await _register(client, "me@test.com")
        token = reg.json()["access_token"]
        r = await client.get("/auth/me", headers=_auth_header(token))
        assert r.status_code == 200
        assert r.json()["email"] == "me@test.com"

    async def test_me_with_garbage_token(self, client):
        r = await client.get("/auth/me", headers=_auth_header("garbage.token.value"))
        assert r.status_code == 401


# ── repos ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRepos:
    async def test_repos_requires_auth(self, client):
        r = await client.get("/repos")
        assert r.status_code == 403

    async def test_repos_empty_for_new_user(self, client):
        reg = await _register(client, "repos@test.com")
        token = reg.json()["access_token"]
        r = await client.get("/repos", headers=_auth_header(token))
        assert r.status_code == 200
        assert r.json()["repos"] == []


# ── chat ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestChat:
    async def test_chat_requires_auth(self, client):
        r = await client.post("/chat", json={"query": "hi", "repo_id": "x"})
        assert r.status_code == 403

    async def test_chat_returns_404_for_unindexed_repo(self, client):
        reg = await _register(client, "chat@test.com")
        token = reg.json()["access_token"]
        r = await client.post(
            "/chat",
            json={"query": "how does auth work?", "repo_id": "https://github.com/x/y"},
            headers=_auth_header(token),
        )
        assert r.status_code == 404

    async def test_conversational_query_short_circuits(self, client):
        """Greetings bypass RAG and return a canned response without calling OpenAI."""
        reg = await _register(client, "conv@test.com")
        token = reg.json()["access_token"]
        with patch("main.load_indexed_repos", return_value=["https://github.com/x/y"]), \
             patch("main.get_collection_for_repo", return_value=MagicMock()):
            r = await client.post(
                "/chat",
                json={"query": "hello", "repo_id": "https://github.com/x/y"},
                headers=_auth_header(token),
            )
        assert r.status_code == 200
        assert r.json()["sources"] == []


# ── index ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestIndex:
    async def test_index_requires_auth(self, client):
        r = await client.post("/index", json={"repo_url": "https://github.com/x/y"})
        assert r.status_code == 403

    async def test_index_already_indexed_returns_status(self, client):
        reg = await _register(client, "index@test.com")
        token = reg.json()["access_token"]
        mock_col = MagicMock()
        mock_col.count.return_value = 42
        with patch("main.is_repo_indexed", return_value=True), \
             patch("main.get_collection_for_repo", return_value=mock_col):
            r = await client.post(
                "/index",
                json={"repo_url": "https://github.com/x/y"},
                headers=_auth_header(token),
            )
        assert r.status_code == 200
        assert r.json()["status"] == "already_indexed"
        assert r.json()["chunks_indexed"] == 42
