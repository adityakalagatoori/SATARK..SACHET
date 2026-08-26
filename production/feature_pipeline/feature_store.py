"""
Feature store — keeps training-time and serving-time features consistent.

Why this exists: the prototype extracts features once from a static CSV
(src/satark.py's extract_features()). In production, the same account's
7/14/31-day transaction ratios must be computed identically whether the
model is being retrained offline on historical data or scoring a live
transaction — if the two computations ever drift apart ("train/serve
skew"), the model silently degrades without any code actually breaking.

Design: one function, compute_rolling_features(), is the single source of
truth for turning a list of TransactionEvent objects into the feature
values SACHET's models expect. Both the offline (batch, historical) path
and the online (live, incremental) path call this exact function — they
never each implement their own version of "sum of cash credits in the last
7 days."

Two storage backends: InMemoryFeatureStore (zero dependencies, used for
local development and tests) and RedisFeatureStore (the production choice —
sub-millisecond reads needed for real-time scoring at transaction time,
requires a running Redis instance to actually use, provided here as
correct, ready code).
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import json

from finacle_adapter import TransactionEvent

_WINDOWS_DAYS = (7, 14, 31)


def compute_rolling_features(account_id: str, events: list[TransactionEvent],
                             as_of: datetime = None) -> dict:
    """The single source of truth for account-level rolling features.
    Mirrors the channel x window x statistic structure of the competition
    dataset's F-columns (see data/Description.xlsx) so a production model
    retrained on live-computed features stays compatible with the same
    architecture validated in src/satark.py — total/credit/debit amount and
    count per channel per window, computed the SAME way whether this is
    called for offline training data or a live scoring request."""
    as_of = as_of or datetime.now(timezone.utc)
    account_events = [e for e in events if e.account_id == account_id]

    features = {}
    channels = sorted(set(e.channel for e in account_events)) or ['CASH', 'CHEQUE', 'UPI',
                                                                    'NEFT', 'IMPS', 'RTGS', 'ATM', 'POS']
    for window in _WINDOWS_DAYS:
        cutoff = as_of - timedelta(days=window)
        in_window = [e for e in account_events if cutoff <= e.timestamp <= as_of]
        for channel in channels:
            ch_events = [e for e in in_window if e.channel == channel]
            cr = [e for e in ch_events if e.direction == 'CREDIT']
            db = [e for e in ch_events if e.direction == 'DEBIT']
            prefix = f"{channel}_L{window}D"
            features[f"{prefix}_TOTAL_AMT"] = sum(e.amount for e in ch_events)
            features[f"{prefix}_CR_AMT"] = sum(e.amount for e in cr)
            features[f"{prefix}_DB_AMT"] = sum(e.amount for e in db)
            features[f"{prefix}_TOTAL_TXNS"] = len(ch_events)
            features[f"{prefix}_CR_TXNS"] = len(cr)
            features[f"{prefix}_DB_TXNS"] = len(db)

    # ratio features (7-to-14, 14-to-31) mirroring the R_ column family
    for a, b in [(7, 14), (14, 31)]:
        for channel in channels:
            num = features.get(f"{channel}_L{a}D_TOTAL_AMT", 0)
            den = features.get(f"{channel}_L{b}D_TOTAL_AMT", 0)
            features[f"R_{channel}_AMT_L{a}_{b}D"] = (num / den) if den else 0.0

    features['_computed_at'] = as_of.isoformat()
    features['_account_id'] = account_id
    return features


# ============================================================================
# EXTENDED RAW FEATURES — the fuller channel taxonomy schema_mapper.py needs
# to reconstruct the competition dataset's F-columns from live event data.
# ============================================================================

# The 16 channel categories Description.xlsx's Variable Name / Description
# text actually distinguishes (found by parsing every one of the 3,924
# feature descriptions — see schema_mapper.py) — a superset of the 8-channel
# list compute_rolling_features() above uses for its own, smaller, already-
# tested feature set.
ALL_CHANNELS = ('CASH', 'CHEQUE', 'UPI', 'NEFT', 'IMPS', 'RTGS', 'ATM', 'POS',
                 'STDNG_INSTR', 'APB', 'BBPS', 'MBNKING', 'NET_BNKING', 'GST',
                 'LOAN', 'FEES_CHRGS')

# Two combined categories Description.xlsx tracks that are NOT raw channels —
# they're derived sums over other channels, computed here rather than tagged
# on any single event.
ELEC_XFER_MEMBERS = ('NEFT', 'IMPS', 'RTGS')          # "Online Transfer (IMPS+NEFT+RTGS)"
NON_CASH_CHQ_EXCLUDE = ('CASH', 'CHEQUE')              # "Non Cash Non Cheque" = everything else


def _amount_stats(events_in_window):
    amts = [e.amount for e in events_in_window]
    cr = [e.amount for e in events_in_window if e.direction == 'CREDIT']
    db = [e.amount for e in events_in_window if e.direction == 'DEBIT']
    ci = [e for e in events_in_window if e.is_customer_induced]
    return {
        'total_amt': sum(amts), 'cr_amt': sum(cr), 'db_amt': sum(db),
        'total_txns': len(events_in_window), 'cr_txns': len(cr), 'db_txns': len(db),
        'min_amt': min(amts) if amts else 0.0,
        'max_amt': max(amts) if amts else 0.0,
        'avg_amt': (sum(amts) / len(amts)) if amts else 0.0,
        'ci_total_amt': sum(e.amount for e in ci),
        'ci_total_txns': len(ci),
    }


def _sum_stat_dicts(dicts):
    if not dicts:
        return _amount_stats([])
    out = {k: 0.0 for k in dicts[0]}
    for d in dicts:
        for k, v in d.items():
            out[k] += v
    return out


def compute_extended_rolling_features(account_id: str, events: list, as_of: datetime = None) -> dict:
    """Raw per-channel-per-window aggregate quantities for the FULL 16-channel
    taxonomy, plus derived combined categories and balance-window stats.

    Kept as a SEPARATE function from compute_rolling_features() above (Phase
    2's original, already-tested function with its own smaller 8-channel key
    set) rather than modified in place, so nothing depending on that
    function's exact output breaks. schema_mapper.py consumes this function's
    output to reconstruct the competition dataset's ~3,924 F-columns from
    live event data — see that module for exactly which statistic families
    (ratio, deviation, min/max/avg, ...) are derived from these raw numbers,
    and which conventions are best-effort / need confirmation against BOI's
    real feature-derivation logic before production use.

    Returns a nested dict:
      raw[channel][window_days] = {total_amt, cr_amt, db_amt, total_txns,
                                    cr_txns, db_txns, min_amt, max_amt,
                                    avg_amt, ci_total_amt, ci_total_txns}
      raw['_ALL'][window_days]  = the same, summed across every channel in
                                  ALL_CHANNELS (cross-channel totals)
      raw['NON_CASH_CHQ'][w]    = summed across ALL_CHANNELS minus CASH/CHEQUE
      raw['ELEC_XFER'][w]       = summed across NEFT+IMPS+RTGS
      raw['_BAL'][window_days]  = {min_bal, max_bal, avg_bal, total_bal} from
                                  TransactionEvent.balance_after snapshots
    """
    as_of = as_of or datetime.now(timezone.utc)
    account_events = [e for e in events if e.account_id == account_id]
    windows = (7, 14, 31)

    raw = {}
    for channel in ALL_CHANNELS:
        raw[channel] = {}
        for w in windows:
            cutoff = as_of - timedelta(days=w)
            in_window = [e for e in account_events
                        if cutoff <= e.timestamp <= as_of and e.channel == channel]
            raw[channel][w] = _amount_stats(in_window)

    for w in windows:
        raw.setdefault('_ALL', {})[w] = _sum_stat_dicts([raw[c][w] for c in ALL_CHANNELS])
        raw.setdefault('NON_CASH_CHQ', {})[w] = _sum_stat_dicts(
            [raw[c][w] for c in ALL_CHANNELS if c not in NON_CASH_CHQ_EXCLUDE])
        raw.setdefault('ELEC_XFER', {})[w] = _sum_stat_dicts(
            [raw[c][w] for c in ELEC_XFER_MEMBERS])

        cutoff = as_of - timedelta(days=w)
        bal_snapshots = [e.balance_after for e in account_events
                         if e.balance_after is not None and cutoff <= e.timestamp <= as_of]
        raw.setdefault('_BAL', {})[w] = {
            'min_bal': min(bal_snapshots) if bal_snapshots else 0.0,
            'max_bal': max(bal_snapshots) if bal_snapshots else 0.0,
            'avg_bal': (sum(bal_snapshots) / len(bal_snapshots)) if bal_snapshots else 0.0,
            'total_bal': sum(bal_snapshots),
        }

    return raw


class FeatureStore(ABC):
    @abstractmethod
    def get_online_features(self, account_id: str) -> dict:
        """Low-latency read for live scoring at transaction time."""
        raise NotImplementedError

    @abstractmethod
    def update_online_features(self, account_id: str, features: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_offline_features(self, account_ids: list[str]) -> dict:
        """Batch read for training/retraining — may hit a different, slower
        store than the online path, but must be computed by the SAME
        compute_rolling_features() logic to stay train/serve-consistent."""
        raise NotImplementedError


class InMemoryFeatureStore(FeatureStore):
    """Dict-backed store — zero external dependencies, correct behavior, not
    suitable for multi-process production use (state doesn't survive a
    restart or scale across instances). Used for local development/tests and
    as the reference implementation the Redis backend must match."""

    def __init__(self):
        self._store = {}

    def get_online_features(self, account_id: str) -> dict:
        return self._store.get(account_id, {})

    def update_online_features(self, account_id: str, features: dict) -> None:
        self._store[account_id] = features

    def get_offline_features(self, account_ids: list[str]) -> dict:
        return {aid: self._store.get(aid, {}) for aid in account_ids}


class RedisFeatureStore(FeatureStore):
    """Production backend. Requires a running Redis instance — this class is
    correct, ready-to-run code, not runnable in this sandboxed environment
    without one. Redis is the standard choice here because live fraud
    scoring needs sub-millisecond feature reads at transaction time, which
    an in-memory-per-process store cannot provide once there is more than
    one serving instance."""

    def __init__(self, host='localhost', port=6379, db=0):
        try:
            import redis
        except ImportError:
            raise ImportError("redis not installed — pip install redis")
        self._client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def get_online_features(self, account_id: str) -> dict:
        raw = self._client.get(f"features:{account_id}")
        return json.loads(raw) if raw else {}

    def update_online_features(self, account_id: str, features: dict) -> None:
        self._client.set(f"features:{account_id}", json.dumps(features))

    def get_offline_features(self, account_ids: list[str]) -> dict:
        return {aid: self.get_online_features(aid) for aid in account_ids}
