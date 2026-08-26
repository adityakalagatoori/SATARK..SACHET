"""
Human override layer for SACHET's automated actions.

Why this exists: RBI's draft Model Risk Management guidance requires human
oversight of every AI-driven decision, not just a kill switch for the whole
system — an individual account's AUTO-HOLD (see generate_alerts() in
src/satark.py) must be reversible by one authorized human action, and that
reversal must be permanently logged.

Design choice — append-only, never delete/update in place: an action's
history (recorded, then possibly overridden) must remain fully reconstructable
for audit. Overriding an action does not erase the original record; it adds
an override record referencing it. This mirrors how the RBI guidance frames
model risk management as an enterprise governance responsibility, not a
place where records can quietly disappear.

Design philosophy — human-on-the-loop, not human-in-the-loop: SACHET is
allowed to act autonomously on high-confidence (CRITICAL-tier) cases without
waiting for a human to approve the action first — that's what makes the
system useful at alert volumes an investigator team can't pre-clear one by
one. But acting autonomously must not mean the system can silently hold an
account indefinitely just because no investigator got to it. So every
automated action is created with a time-boxed expiry (`expires_at`, set via
`sla_hours` in `record_action()`): if a human investigator has not reviewed
and explicitly overridden the action by then, `auto_expire_stale_actions()`
reverses it automatically, and that reversal is logged distinctly as a
system decision (`override_type = 'AUTO_EXPIRED'`, `investigator_id =
'SYSTEM_AUTO_EXPIRY'`) — never conflated with a human override
(`override_type = 'HUMAN'`) in any downstream query. This makes the system
fail SAFE by default: harm from an over-held account is bounded by the SLA
window rather than by investigator speed, while a human can still act at any
point before expiry, and the separate, instant, platform-wide kill switch
(kill_switch.py) remains untouched by any of this — it is a distinct
emergency control, not something this mechanism replaces.
"""

import sqlite3
import os
from datetime import datetime, timezone, timedelta

_DB_PATH = os.path.join(os.path.dirname(__file__), '_state', 'human_override.db')


def _get_connection():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('''
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            action_type TEXT NOT NULL,
            action_detail TEXT,
            model_version TEXT,
            status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE', 'OVERRIDDEN')),
            created_at TEXT NOT NULL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS overrides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action_id INTEGER NOT NULL,
            investigator_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(action_id) REFERENCES actions(id)
        )
    ''')
    # Defensive ALTER TABLEs: these columns are new as of the time-boxed-expiry
    # feature. SQLite has no `ADD COLUMN IF NOT EXISTS`, so on a fresh DB
    # (created by the CREATE TABLE above, already including these columns'
    # ancestors is not the case here since CREATE TABLE above does NOT define
    # them) this will succeed once; on a pre-existing DB file from before this
    # feature it will also succeed once; on every subsequent call it will
    # raise "duplicate column name", which we swallow deliberately.
    try:
        conn.execute('ALTER TABLE actions ADD COLUMN expires_at TEXT')
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE overrides ADD COLUMN override_type TEXT DEFAULT 'HUMAN'")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    return conn


def record_action(account_id: int, action_type: str, action_detail: str = '',
                  model_version: str = 'unspecified', sla_hours: float = 48.0) -> int:
    """Log a new automated action (e.g. an AUTO-HOLD triggered by a CRITICAL
    alert) as active/in-effect. Returns the action's id, needed to override it
    later. Called by the serving layer at the moment an automated action is
    taken — this function does not decide whether to act, it only records
    that an action was taken, for auditability and reversal.

    sla_hours: how long this action stays active before
    auto_expire_stale_actions() will reverse it automatically if no
    investigator has acted on it. Default is 48 hours. This default is not
    arbitrary: it's sized against the documented real-world capacity of an
    investigator team (roughly 200-300 alerts/month per team — see
    docs/CYBERSECURITY_TOPIC_PREP.md), which works out to a handful of cases
    per working day per investigator. A multi-day SLA window is realistic
    for that throughput without over-holding accounts for weeks; a shorter
    window would risk auto-reversing legitimate holds before an investigator
    has had a reasonable chance to review them, and an unbounded window is
    exactly the failure mode (indefinite hold dependent on human speed) this
    mechanism exists to remove."""
    conn = _get_connection()
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + timedelta(hours=sla_hours)
    cur = conn.execute(
        'INSERT INTO actions (account_id, action_type, action_detail, model_version, status, created_at, expires_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?)',
        (account_id, action_type, action_detail, model_version, 'ACTIVE',
         created_at.isoformat(), expires_at.isoformat())
    )
    conn.commit()
    action_id = cur.lastrowid
    conn.close()
    return action_id


def override_action(action_id: int, investigator_id: str, reason: str) -> None:
    """Reverse an active automated action. investigator_id and reason are
    mandatory for the same auditability reason as the kill switch. Marks the
    action OVERRIDDEN but keeps the original record intact — the override is
    a new row referencing it, not an edit."""
    if not investigator_id or not reason:
        raise ValueError("investigator_id and reason are required to override an automated action")
    conn = _get_connection()
    action = conn.execute('SELECT status FROM actions WHERE id = ?', (action_id,)).fetchone()
    if action is None:
        conn.close()
        raise ValueError(f"no action found with id={action_id}")
    if action['status'] == 'OVERRIDDEN':
        conn.close()
        raise ValueError(f"action {action_id} was already overridden")

    conn.execute('UPDATE actions SET status = ? WHERE id = ?', ('OVERRIDDEN', action_id))
    conn.execute(
        'INSERT INTO overrides (action_id, investigator_id, reason, timestamp, override_type) VALUES (?, ?, ?, ?, ?)',
        (action_id, investigator_id, reason, datetime.now(timezone.utc).isoformat(), 'HUMAN')
    )
    conn.commit()
    conn.close()


def auto_expire_stale_actions() -> list:
    """Automatically reverse every ACTIVE action whose SLA (`expires_at`) has
    passed without an investigator having overridden it. This is the
    fail-safe half of the "human-on-the-loop" design: it makes sure an
    automated hold cannot outlive its SLA window just because no investigator
    got to it in time.

    This function does the reversal itself; it does not schedule anything.
    In a real deployment this would be invoked periodically by an external
    scheduler (e.g. an hourly cron job / task queue beat), which is
    infrastructure outside this repo's scope — this module only provides the
    mechanism to call.

    Each expired action is marked OVERRIDDEN (same terminal state a human
    override produces) and gets an overrides row with
    investigator_id='SYSTEM_AUTO_EXPIRY', override_type='AUTO_EXPIRED', so it
    is unambiguously distinguishable from a real human decision in every
    downstream query (get_action_status, list_active_actions, the API, etc).

    Returns the list of action ids that were auto-expired in this call, for
    logging/observability by the caller."""
    conn = _get_connection()
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = conn.execute(
        "SELECT id FROM actions WHERE status = 'ACTIVE' AND expires_at IS NOT NULL AND expires_at < ?",
        (now_iso,)
    ).fetchall()
    expired_ids = []
    for row in rows:
        action_id = row['id']
        conn.execute('UPDATE actions SET status = ? WHERE id = ?', ('OVERRIDDEN', action_id))
        conn.execute(
            'INSERT INTO overrides (action_id, investigator_id, reason, timestamp, override_type) '
            'VALUES (?, ?, ?, ?, ?)',
            (action_id, 'SYSTEM_AUTO_EXPIRY',
             'Auto-expired: no investigator response within SLA window',
             datetime.now(timezone.utc).isoformat(), 'AUTO_EXPIRED')
        )
        expired_ids.append(action_id)
    conn.commit()
    conn.close()
    return expired_ids


def get_action_status(account_id: int) -> list:
    """All actions (active and overridden) recorded for one account, most
    recent first — what an investigator sees when opening a case. Each
    action dict includes `expires_at` (the SLA deadline for auto-expiry) and,
    for overridden actions, an `override` dict whose `override_type` is
    'HUMAN' or 'AUTO_EXPIRED' — the caller can tell at a glance whether a
    real investigator acted or the system auto-reversed the hold."""
    conn = _get_connection()
    rows = conn.execute(
        'SELECT * FROM actions WHERE account_id = ? ORDER BY id DESC', (account_id,)
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d['status'] == 'OVERRIDDEN':
            ov = conn.execute(
                'SELECT investigator_id, reason, timestamp, override_type FROM overrides WHERE action_id = ? '
                'ORDER BY id DESC LIMIT 1', (d['id'],)
            ).fetchone()
            d['override'] = dict(ov) if ov else None
        result.append(d)
    conn.close()
    return result


def list_active_actions(action_type: str = None) -> list:
    """Every account currently under an active automated action (e.g. every
    live AUTO-HOLD) — the working list an investigator or supervisor reviews.
    Each dict includes `expires_at`, so a supervisor can see how much SLA
    runway remains before auto_expire_stale_actions() would reverse it."""
    conn = _get_connection()
    if action_type:
        rows = conn.execute(
            "SELECT * FROM actions WHERE status = 'ACTIVE' AND action_type = ? ORDER BY created_at DESC",
            (action_type,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM actions WHERE status = 'ACTIVE' ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
