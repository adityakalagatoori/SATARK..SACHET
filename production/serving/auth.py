"""
Authentication and role-based access control (RBAC).

Why this exists: the prototype's serve_api() (src/satark.py) has no
authentication at all — anyone who can reach the port can call /predict.
A production API touching account risk scores and transaction holds needs
both identity (who is calling) and authorization (what they're allowed to
do) — and RBI's guidance on human oversight implies specific roles: an
INVESTIGATOR should be able to view/override alerts, but only an ADMIN
should be able to touch the kill switch, and an AUDITOR needs read-only
access to everything for independent review, per the board-level-
accountability expectation.

Design: API keys map to roles, not to individual permissions — permissions
are derived from role via ROLE_PERMISSIONS, so adding a new role or endpoint
means updating one table, not hunting through every handler.
"""

import sqlite3
import os
import hashlib
import secrets
from datetime import datetime, timezone
from functools import wraps

_DB_PATH = os.path.join(os.path.dirname(__file__), '_state', 'auth.db')

ROLES = ('INVESTIGATOR', 'ADMIN', 'AUDITOR')

ROLE_PERMISSIONS = {
    'INVESTIGATOR': {'view_alerts', 'override_action', 'view_explanation'},
    'ADMIN': {'view_alerts', 'override_action', 'view_explanation',
              'kill_switch_activate', 'kill_switch_deactivate', 'manage_users',
              'view_kill_switch_history', 'view_override_history', 'view_audit_log'},
    'AUDITOR': {'view_alerts', 'view_explanation', 'view_audit_log',
                'view_kill_switch_history', 'view_override_history'},
}


class AuthorizationError(Exception):
    pass


def _get_connection():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('''
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash TEXT NOT NULL UNIQUE,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('INVESTIGATOR', 'ADMIN', 'AUDITOR')),
            created_at TEXT NOT NULL,
            revoked INTEGER NOT NULL DEFAULT 0
        )
    ''')
    conn.commit()
    return conn


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode('utf-8')).hexdigest()


def issue_api_key(user_id: str, role: str) -> str:
    """Generate and store a new API key. Returns the RAW key exactly once —
    only its hash is stored, matching how credentials should be handled;
    losing the raw key means issuing a new one, not recovering the old one."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    raw_key = secrets.token_urlsafe(32)
    conn = _get_connection()
    conn.execute(
        'INSERT INTO api_keys (key_hash, user_id, role, created_at, revoked) VALUES (?, ?, ?, ?, 0)',
        (_hash_key(raw_key), user_id, role, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()
    return raw_key


def revoke_api_key(raw_key: str) -> None:
    conn = _get_connection()
    conn.execute('UPDATE api_keys SET revoked = 1 WHERE key_hash = ?', (_hash_key(raw_key),))
    conn.commit()
    conn.close()


def authenticate(raw_key: str) -> dict:
    """Returns {'user_id', 'role'} for a valid, non-revoked key, or raises
    AuthorizationError."""
    conn = _get_connection()
    row = conn.execute(
        'SELECT user_id, role FROM api_keys WHERE key_hash = ? AND revoked = 0',
        (_hash_key(raw_key),)
    ).fetchone()
    conn.close()
    if row is None:
        raise AuthorizationError("invalid or revoked API key")
    return dict(row)


def authorize(role: str, permission: str) -> None:
    """Raises AuthorizationError if `role` does not have `permission`."""
    if permission not in ROLE_PERMISSIONS.get(role, set()):
        raise AuthorizationError(f"role {role!r} does not have permission {permission!r}")


def require_permission(permission: str):
    """Decorator for Flask-style route handlers. Expects the request to
    carry an 'X-API-Key' header; on success, injects the authenticated
    user's identity as `caller` kwarg into the wrapped function."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            from flask import request, jsonify
            raw_key = request.headers.get('X-API-Key')
            if not raw_key:
                return jsonify({'error': 'missing X-API-Key header'}), 401
            try:
                caller = authenticate(raw_key)
                authorize(caller['role'], permission)
            except AuthorizationError as e:
                return jsonify({'error': str(e)}), 403
            return fn(*args, caller=caller, **kwargs)
        return wrapper
    return decorator
