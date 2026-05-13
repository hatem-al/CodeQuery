"""Unit tests for password hashing, JWT tokens, and user management."""

import pytest
from datetime import timedelta

from auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    decode_access_token,
    create_user,
    authenticate_user,
    get_user_by_email,
)


class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        hashed = get_password_hash("securepass1")
        assert hashed != "securepass1"
        assert len(hashed) > 20

    def test_correct_password_verifies(self):
        hashed = get_password_hash("securepass1")
        assert verify_password("securepass1", hashed) is True

    def test_wrong_password_fails(self):
        hashed = get_password_hash("securepass1")
        assert verify_password("wrongpassword", hashed) is False

    def test_different_passwords_produce_different_hashes(self):
        h1 = get_password_hash("password1")
        h2 = get_password_hash("password2")
        assert h1 != h2

    def test_same_password_different_salts(self):
        h1 = get_password_hash("samepassword1")
        h2 = get_password_hash("samepassword1")
        # bcrypt uses random salt — hashes should differ
        assert h1 != h2


class TestJWTTokens:
    def test_token_encodes_subject(self):
        token = create_access_token({"sub": "user_42"})
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "user_42"

    def test_token_includes_expiry(self):
        token = create_access_token({"sub": "user_42"})
        payload = decode_access_token(token)
        assert "exp" in payload

    def test_expired_token_returns_none(self):
        token = create_access_token({"sub": "user_42"}, expires_delta=timedelta(seconds=-1))
        assert decode_access_token(token) is None

    def test_tampered_token_returns_none(self):
        token = create_access_token({"sub": "user_42"})
        tampered = token[:-8] + "XXXXXXXX"
        assert decode_access_token(tampered) is None

    def test_garbage_string_returns_none(self):
        assert decode_access_token("not.a.token") is None

    def test_custom_expiry_respected(self):
        token = create_access_token({"sub": "u"}, expires_delta=timedelta(hours=1))
        payload = decode_access_token(token)
        assert payload is not None


class TestUserManagement:
    def test_create_user_returns_user_dict(self):
        user = create_user("alice@example.com", "password1")
        assert user["email"] == "alice@example.com"
        assert "id" in user
        assert "hashed_password" in user

    def test_created_user_is_retrievable(self):
        create_user("bob@example.com", "password1")
        user = get_user_by_email("bob@example.com")
        assert user is not None
        assert user["email"] == "bob@example.com"

    def test_duplicate_email_raises(self):
        create_user("dup@example.com", "password1")
        with pytest.raises(ValueError, match="already exists"):
            create_user("dup@example.com", "password1")

    def test_authenticate_valid_credentials(self):
        create_user("carol@example.com", "password1")
        user = authenticate_user("carol@example.com", "password1")
        assert user is not None
        assert user["email"] == "carol@example.com"

    def test_authenticate_wrong_password(self):
        create_user("dave@example.com", "password1")
        assert authenticate_user("dave@example.com", "wrongpass") is None

    def test_authenticate_nonexistent_user(self):
        assert authenticate_user("nobody@example.com", "password1") is None

    def test_username_defaults_to_email_prefix(self):
        user = create_user("eve@example.com", "password1")
        assert user["username"] == "eve"
