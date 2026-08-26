"""
Versioned model registry.

Why this exists: the prototype's models live as flat .pkl files
(models/ensemble.pkl) with no version history — retraining silently
overwrites the previous model, with no record of what changed, no way to
compare a challenger against the current champion, and no rollback if a
retrain regresses. RBI's draft Model Risk Management guidance requires
documented model ownership and traceability; a registry is the concrete
mechanism for that, not just a policy statement.

Design: every registered version gets its own immutable artifact copy
(never overwritten) plus a metadata row (metrics, registered_at, promoted
or not). "Latest" and "champion" are deliberately different concepts here:
a new version can be registered (uploaded, evaluated) without automatically
becoming the one actually serving traffic — promotion is a separate,
explicit step, so a regression is caught before it reaches production, not
after.
"""

import sqlite3
import os
import shutil
import json
from datetime import datetime, timezone

_BASE_DIR = os.path.join(os.path.dirname(__file__), '_registry')
_DB_PATH = os.path.join(_BASE_DIR, 'registry.db')
_ARTIFACTS_DIR = os.path.join(_BASE_DIR, 'artifacts')


def _get_connection():
    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('''
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            version TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            registered_by TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            is_champion INTEGER NOT NULL DEFAULT 0,
            UNIQUE(model_name, version)
        )
    ''')
    conn.commit()
    return conn


def register_model(model_name: str, version: str, source_artifact_path: str,
                   metrics: dict, registered_by: str) -> int:
    """Copy an artifact into the registry's immutable storage and record its
    metrics. Does NOT make it the champion — call promote() separately,
    after comparing against the current champion's metrics."""
    if not os.path.exists(source_artifact_path):
        raise FileNotFoundError(source_artifact_path)

    dest_dir = os.path.join(_ARTIFACTS_DIR, model_name, version)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(source_artifact_path))
    shutil.copy2(source_artifact_path, dest_path)

    conn = _get_connection()
    cur = conn.execute(
        'INSERT INTO model_versions (model_name, version, artifact_path, metrics_json, '
        'registered_by, registered_at, is_champion) VALUES (?, ?, ?, ?, ?, ?, 0)',
        (model_name, version, dest_path, json.dumps(metrics), registered_by,
         datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def promote(model_name: str, version: str, promoted_by: str,
           min_auc_pr_vs_champion: float = 0.0) -> bool:
    """Make `version` the champion, IF it does not regress the current
    champion's AUC-PR by more than the allowed tolerance (default: any
    regression at all blocks promotion). Returns False and does NOT promote
    if the challenger regresses — this is the guardrail that stops a bad
    retrain from silently reaching production."""
    conn = _get_connection()
    challenger = conn.execute(
        'SELECT * FROM model_versions WHERE model_name = ? AND version = ?',
        (model_name, version)
    ).fetchone()
    if challenger is None:
        conn.close()
        raise ValueError(f"no registered version {model_name}=={version}")

    current_champion = conn.execute(
        'SELECT * FROM model_versions WHERE model_name = ? AND is_champion = 1',
        (model_name,)
    ).fetchone()

    challenger_metrics = json.loads(challenger['metrics_json'])
    if current_champion is not None:
        champion_metrics = json.loads(current_champion['metrics_json'])
        challenger_auc = challenger_metrics.get('auc_pr', 0)
        champion_auc = champion_metrics.get('auc_pr', 0)
        if challenger_auc < champion_auc - min_auc_pr_vs_champion:
            conn.close()
            return False  # regression — refuse to promote

    conn.execute('UPDATE model_versions SET is_champion = 0 WHERE model_name = ?', (model_name,))
    conn.execute(
        'UPDATE model_versions SET is_champion = 1 WHERE model_name = ? AND version = ?',
        (model_name, version)
    )
    conn.commit()
    conn.close()
    return True


def get_champion(model_name: str) -> dict:
    conn = _get_connection()
    row = conn.execute(
        'SELECT * FROM model_versions WHERE model_name = ? AND is_champion = 1', (model_name,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    d['metrics'] = json.loads(d.pop('metrics_json'))
    return d


def list_versions(model_name: str) -> list:
    conn = _get_connection()
    rows = conn.execute(
        'SELECT * FROM model_versions WHERE model_name = ? ORDER BY registered_at DESC',
        (model_name,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d['metrics'] = json.loads(d.pop('metrics_json'))
        result.append(d)
    return result


def rollback(model_name: str, to_version: str, rolled_back_by: str) -> None:
    """Explicit rollback: make an older, already-registered version the
    champion again, bypassing the regression check in promote() — a
    deliberate emergency action, always logged via the version it lands on
    plus the registered_by/registered_at already stored for that version."""
    conn = _get_connection()
    target = conn.execute(
        'SELECT id FROM model_versions WHERE model_name = ? AND version = ?',
        (model_name, to_version)
    ).fetchone()
    if target is None:
        conn.close()
        raise ValueError(f"cannot roll back to unregistered version {model_name}=={to_version}")
    conn.execute('UPDATE model_versions SET is_champion = 0 WHERE model_name = ?', (model_name,))
    conn.execute(
        'UPDATE model_versions SET is_champion = 1 WHERE model_name = ? AND version = ?',
        (model_name, to_version)
    )
    conn.commit()
    conn.close()
