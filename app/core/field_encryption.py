"""Authenticated encryption for Agent-domain sensitive payloads."""
from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings


def _cipher() -> Fernet:
    if not settings.FIELD_ENCRYPTION_KEY:
        raise RuntimeError("FIELD_ENCRYPTION_KEY must be configured for internal-note storage")
    return Fernet(settings.FIELD_ENCRYPTION_KEY.encode("ascii"))


def encrypt_internal_note(value: str) -> str:
    return _cipher().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_internal_note(value: str) -> str:
    try:
        return _cipher().decrypt(value.encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeEncodeError) as exc:
        raise ValueError("Internal note ciphertext is invalid") from exc
