"""
Kill switch for SACHET's automated decisioning.

Why this exists: RBI's draft "Guidance on Regulatory Principles for Model
Risk Management, 2026" mandates an emergency kill switch for every AI/ML
model used in a regulated financial entity — a mechanism to instantly
deactivate model-driven decisions, independent of the model's own code,
that any authorized person can trigger without needing engineering
involvement.

When active, every caller in the serving layer MUST check is_active() before
taking any automated action (e.g. the AUTO-HOLD recommendation from
generate_alerts() in src/satark.py) and fall back to the bank's pre-existing
rule-based system instead. This module does not implement that fallback
itself — it only provides the switch and its audit trail; every calling
service is responsible for checking it.

State is persisted to SQLite (not just in-memory) so the switch survives a
process restart, and every state change is logged immutably — RBI's guidance
explicitly requires documented model ownership and traceability of who
changed what and why, not just a boolean flag.
"""

import sqlite3
import os
from datetime import datetime, timezone

_DB_PATH = os.path.join(os.path.dirname(__file__), '_state', 'kill_switch.db')


def _get_connection():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('''
        CREATE TABLE IF NOT EXISTS kill_switch_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL CHECK(action IN ('ACTIVATE', 'DEACTIVATE')),
            triggered_by TEXT NOT NULL,
            reason TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn


def activate(triggered_by: str, reason: str) -> None:
    """Immediately halt all automated model decisions. triggered_by and
    reason are mandatory — RBI guidance requires an auditable justification
    for every governance action, not a bare toggle."""
    if not triggered_by or not reason:
        raise ValueError("triggered_by and reason are required for an auditable kill-switch action")
    conn = _get_connection()
    conn.execute(
        'INSERT INTO kill_switch_events (action, triggered_by, reason, timestamp) VALUES (?, ?, ?, ?)',
        ('ACTIVATE', triggered_by, reason, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()


def deactivate(triggered_by: str, reason: str) -> None:
    """Resume automated model decisions. Same auditability requirement as activate()."""
    if not triggered_by or not reason:
        raise ValueError("triggered_by and reason are required for an auditable kill-switch action")
    conn = _get_connection()
    conn.execute(
        'INSERT INTO kill_switch_events (action, triggered_by, reason, timestamp) VALUES (?, ?, ?, ?)',
        ('DEACTIVATE', triggered_by, reason, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()


def is_active() -> bool:
    """True if the kill switch is currently engaged (model decisions must be
    suppressed). The switch state is derived from the most recent event, not
    a separately-stored flag, so it can never drift out of sync with the
    audit log."""
    conn = _get_connection()
    row = conn.execute(
        'SELECT action FROM kill_switch_events ORDER BY id DESC LIMIT 1'
    ).fetchone()
    conn.close()
    return row is not None and row['action'] == 'ACTIVATE'


def get_history(limit: int = 100) -> list:
    """Full auditable history of every activation/deactivation, most recent first."""
    conn = _get_connection()
    rows = conn.execute(
        'SELECT action, triggered_by, reason, timestamp FROM kill_switch_events '
        'ORDER BY id DESC LIMIT ?', (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_status() -> dict:
    """Convenience summary: current state plus who/when it was last changed."""
    history = get_history(limit=1)
    return {
        'active': is_active(),
        'last_change': history[0] if history else None,
    }
