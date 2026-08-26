"""Regression tests for Phase 2 — core banking adapter and feature store."""

import pytest
from datetime import datetime, timezone, timedelta

from finacle_adapter import MockCoreBankingAdapter, TransactionEvent
from feature_store import compute_rolling_features, InMemoryFeatureStore


def test_mock_adapter_produces_valid_events():
    adapter = MockCoreBankingAdapter(n_accounts=5, seed=1)
    events = adapter.fetch_new_transactions(since=datetime.now(timezone.utc) - timedelta(days=1))
    assert len(events) > 0
    for e in events:
        assert isinstance(e, TransactionEvent)
        assert e.direction in ('CREDIT', 'DEBIT')
        assert e.amount > 0


def test_compute_rolling_features_windows_are_monotonic_in_scope():
    """A 31-day window must include everything a 7-day window includes for
    the same account — if this breaks, the windowing logic itself is wrong,
    not just imprecise."""
    now = datetime.now(timezone.utc)
    events = [
        TransactionEvent('ACC1', now - timedelta(days=3), 'CASH', 'CREDIT', 1000),
        TransactionEvent('ACC1', now - timedelta(days=20), 'CASH', 'CREDIT', 5000),
    ]
    feats = compute_rolling_features('ACC1', events, as_of=now)
    assert feats['CASH_L7D_TOTAL_AMT'] == 1000       # only the 3-day-old txn
    assert feats['CASH_L31D_TOTAL_AMT'] == 6000       # both txns


def test_compute_rolling_features_isolates_accounts():
    now = datetime.now(timezone.utc)
    events = [
        TransactionEvent('ACC1', now, 'CASH', 'CREDIT', 1000),
        TransactionEvent('ACC2', now, 'CASH', 'CREDIT', 999999),
    ]
    feats = compute_rolling_features('ACC1', events, as_of=now)
    assert feats['CASH_L7D_TOTAL_AMT'] == 1000  # ACC2's huge transaction must not leak in


def test_feature_store_online_offline_consistency():
    """The offline batch read must return exactly what was written via the
    online path — this is the train/serve-consistency guarantee the whole
    module exists for."""
    store = InMemoryFeatureStore()
    feats = {'CASH_L7D_TOTAL_AMT': 5000}
    store.update_online_features('ACC1', feats)
    assert store.get_online_features('ACC1') == feats
    assert store.get_offline_features(['ACC1'])['ACC1'] == feats


def test_feature_store_unknown_account_returns_empty():
    store = InMemoryFeatureStore()
    assert store.get_online_features('NEVER_SEEN') == {}
