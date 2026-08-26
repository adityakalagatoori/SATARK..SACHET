"""
Application-level field encryption for sensitive audit data.

Why this exists: the prototype's audit tables (kill_switch.py,
human_override.py) store investigator reasons and action details as plain
text in SQLite. A regulated bank's audit trail — containing why an account
was held, an investigator's reasoning, customer-identifying context — needs
encryption at rest. Full-database encryption (e.g. SQLCipher) requires a
separate native extension that may not be available in every deployment
environment; application-level field encryption using Fernet (symmetric,
authenticated encryption from the `cryptography` library, already a project
dependency) works anywhere Python runs and is applied selectively to the
sensitive text fields specifically, not the whole database.

Key management note: get_or_create_key() generates and persists a local key
file for development convenience. In actual production this key MUST come
from a proper secrets manager (e.g. HashiCorp Vault, AWS/Azure KMS, or the
bank's existing key-management infrastructure) — a key file sitting next to
the database it encrypts defeats the purpose for anyone with filesystem
access, and this module says so explicitly rather than pretending the local
file is sufficient.
"""

import os
from cryptography.fernet import Fernet

_KEY_PATH = os.path.join(os.path.dirname(__file__), '_state', 'encryption.key')


def get_or_create_key() -> bytes:
    """DEVELOPMENT-ONLY key storage. Production deployments must replace
    this with a call to a real secrets manager / KMS — see module
    docstring. Kept simple here so the encryption logic itself is usable
    and testable without external infrastructure."""
    os.makedirs(os.path.dirname(_KEY_PATH), exist_ok=True)
    if os.path.exists(_KEY_PATH):
        with open(_KEY_PATH, 'rb') as f:
            return f.read()
    key = Fernet.generate_key()
    with open(_KEY_PATH, 'wb') as f:
        f.write(key)
    return key


def encrypt_field(plaintext: str, key: bytes = None) -> str:
    """Encrypt one text field. Returns a string safe to store in a TEXT
    column. key defaults to get_or_create_key() but should be passed
    explicitly once real key management is wired up."""
    key = key or get_or_create_key()
    f = Fernet(key)
    return f.encrypt(plaintext.encode('utf-8')).decode('utf-8')


def decrypt_field(ciphertext: str, key: bytes = None) -> str:
    key = key or get_or_create_key()
    f = Fernet(key)
    return f.decrypt(ciphertext.encode('utf-8')).decode('utf-8')
