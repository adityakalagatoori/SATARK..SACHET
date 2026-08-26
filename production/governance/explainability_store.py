"""
Persistent SHAP explanation storage.

Why this exists: RBI's draft Model Risk Management guidance requires every
AI-driven decision to be explainable on demand — for a regulator, an
auditor, or a customer disputing an action months later. The prototype's
explain_prediction() (src/satark.py) computes real SHAP values but only
prints them to the console; nothing is retrievable after the process exits.
This module fixes that: every explanation computed is written to durable
storage, retrievable by account and by time, so "why was this flagged" has a
permanent answer, not a one-time console log that's already gone.

This mirrors the SHAP TreeExplainer logic in src/satark.py's
explain_prediction() but is hardened for production: results are persisted,
versioned by model_version, and queryable by account and time range instead
of being computed fresh (and lost) on every call.
"""

import sqlite3
import os
import json
from datetime import datetime, timezone

_DB_PATH = os.path.join(os.path.dirname(__file__), '_state', 'explanations.db')


def _get_connection():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('''
        CREATE TABLE IF NOT EXISTS explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            mule_score REAL,
            top_features_json TEXT NOT NULL,
            computed_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn


def save_explanation(account_id: int, model_version: str, top_features: list,
                     mule_score: float = None) -> int:
    """Persist one explanation. top_features is a list of
    {'feature': str, 'shap_value': float, 'direction': 'increases'|'decreases'}
    dicts, ordered by |shap_value| descending — the same shape
    explain_prediction() already produces internally in src/satark.py, just
    written to disk here instead of only printed."""
    conn = _get_connection()
    cur = conn.execute(
        'INSERT INTO explanations (account_id, model_version, mule_score, top_features_json, computed_at) '
        'VALUES (?, ?, ?, ?, ?)',
        (account_id, model_version, mule_score, json.dumps(top_features),
         datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    explanation_id = cur.lastrowid
    conn.close()
    return explanation_id


def compute_and_store_explanation(account_id: int, account_features, model,
                                  feature_names: list, model_version: str,
                                  mule_score: float = None, top_k: int = 10) -> dict:
    """Compute a SHAP TreeExplainer explanation for one account's feature row
    and persist it in one step — the production entry point equivalent to
    src/satark.py's explain_prediction(), but the result survives the
    process. account_features must be a single-row 2D array/DataFrame
    matching feature_names' order."""
    try:
        import shap
    except ImportError:
        raise ImportError("shap not installed — pip install shap")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(account_features)
    raw = shap_values[0] if hasattr(shap_values, '__len__') and len(shap_values) > 0 else shap_values

    pairs = list(zip(feature_names, raw))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    top_features = [
        {'feature': f, 'shap_value': float(v), 'direction': 'increases' if v > 0 else 'decreases'}
        for f, v in pairs[:top_k]
    ]
    save_explanation(account_id, model_version, top_features, mule_score)
    return {'account_id': account_id, 'model_version': model_version,
            'mule_score': mule_score, 'top_features': top_features}


def get_explanation(account_id: int) -> dict:
    """Most recent explanation for an account — what an investigator sees
    when they ask "why is this flagged"."""
    conn = _get_connection()
    row = conn.execute(
        'SELECT * FROM explanations WHERE account_id = ? ORDER BY id DESC LIMIT 1',
        (account_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    d['top_features'] = json.loads(d.pop('top_features_json'))
    return d


def get_explanation_history(account_id: int) -> list:
    """Every explanation ever computed for an account, most recent first —
    lets an auditor see how the model's reasoning for this account changed
    across retrains/model versions over time."""
    conn = _get_connection()
    rows = conn.execute(
        'SELECT * FROM explanations WHERE account_id = ? ORDER BY id DESC', (account_id,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d['top_features'] = json.loads(d.pop('top_features_json'))
        result.append(d)
    return result
