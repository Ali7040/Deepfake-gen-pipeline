"""Password hashing (stdlib PBKDF2) and JWT issuing/verification.

We use `hashlib.pbkdf2_hmac` rather than passlib/bcrypt to avoid native build
dependencies on Windows/Python 3.13 while staying cryptographically sound.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from app.config import settings

_PBKDF2_ROUNDS = 260_000
_ALGO = "sha256"


# ── Password hashing ─────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac(_ALGO, password.encode(), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_{_ALGO}${_PBKDF2_ROUNDS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, rounds_s, salt_hex, hash_hex = stored.split("$")
        rounds = int(rounds_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False
    dk = hashlib.pbkdf2_hmac(_ALGO, password.encode(), salt, rounds)
    return hmac.compare_digest(dk, expected)


# ── JWT ──────────────────────────────────────────────────────────────────────
def _create_token(subject: str, ttl: timedelta, token_type: str) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "type": token_type,
        "iat": now,
        "exp": now + ttl,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(subject: str | int) -> str:
    return _create_token(
        str(subject), timedelta(minutes=settings.access_token_ttl_min), "access"
    )


def create_refresh_token(subject: str | int) -> str:
    return _create_token(
        str(subject), timedelta(days=settings.refresh_token_ttl_days), "refresh"
    )


def decode_token(token: str, expected_type: str | None = None) -> dict[str, Any]:
    """Decode & validate a JWT. Raises jwt.PyJWTError on failure."""
    payload = jwt.decode(
        token, settings.secret_key, algorithms=[settings.jwt_algorithm]
    )
    if expected_type and payload.get("type") != expected_type:
        raise jwt.InvalidTokenError(f"Expected {expected_type} token")
    return payload
