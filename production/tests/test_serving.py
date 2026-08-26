"""Regression tests for Phase 3 (event bus, transaction consumer) and
Phase 5 (auth/RBAC, API)."""

import pytest
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'serving'))

from event_bus import InMemoryEventBus
from transaction_consumer import TransactionConsumer
from feature_store import InMemoryFeatureStore
import kill_switch
from auth import issue_api_key, authenticate, authorize, AuthorizationError, ROLE_PERMISSIONS
from api import app


def _high_score_scorer(features):
    return 0.97 if features.get('CASH_L7D_TOTAL_AMT', 0) > 200000 else 0.1


def test_event_bus_delivers_to_subscriber():
    bus = InMemoryEventBus()
    received = []
    bus.subscribe('topic1', lambda msg: received.append(msg))
    bus.publish('topic1', {'x': 1})
    assert received == [{'x': 1}]


def test_transaction_consumer_triggers_auto_hold_above_threshold():
    bus = InMemoryEventBus()
    consumer = TransactionConsumer(feature_store=InMemoryFeatureStore(), scorer=_high_score_scorer)
    bus.subscribe('transactions', consumer.handle_transaction)
    bus.publish('transactions', {
        'account_id': 'MOCK000001', 'timestamp': datetime.now(timezone.utc).isoformat(),
        'channel': 'CASH', 'direction': 'CREDIT', 'amount': 500000,
    })
    assert consumer.event_log[-1]['action'] == 'AUTO-HOLD'
    assert 'customer_notice' in consumer.event_log[-1]


def test_transaction_consumer_below_threshold_takes_no_action():
    bus = InMemoryEventBus()
    consumer = TransactionConsumer(feature_store=InMemoryFeatureStore(), scorer=_high_score_scorer)
    bus.subscribe('transactions', consumer.handle_transaction)
    bus.publish('transactions', {
        'account_id': 'MOCK000002', 'timestamp': datetime.now(timezone.utc).isoformat(),
        'channel': 'UPI', 'direction': 'DEBIT', 'amount': 500,
    })
    assert consumer.event_log[-1]['action'] is None


def test_transaction_consumer_respects_kill_switch():
    """The specific scenario that matters most: an account that WOULD be
    auto-held must be suppressed instead while the kill switch is active."""
    bus = InMemoryEventBus()
    consumer = TransactionConsumer(feature_store=InMemoryFeatureStore(), scorer=_high_score_scorer)
    bus.subscribe('transactions', consumer.handle_transaction)

    kill_switch.activate(triggered_by='admin', reason='test')
    bus.publish('transactions', {
        'account_id': 'MOCK000003', 'timestamp': datetime.now(timezone.utc).isoformat(),
        'channel': 'CASH', 'direction': 'CREDIT', 'amount': 500000,
    })
    assert consumer.event_log[-1]['action'] == 'SUPPRESSED_BY_KILL_SWITCH'


def test_transaction_consumer_cross_bank_signal_reaches_scoring():
    """A cross-bank flag (DPIP/I4C signal, mocked here) must actually flow
    through feature merging into the score, and a flagged account must
    trigger the same AUTO-HOLD/governance path as any other CRITICAL score
    — this is the concrete integration for the previously-identified
    blind-spot mules (see cross_bank_adapter.py)."""
    from cross_bank_adapter import MockCrossBankAdapter

    def scorer(features):
        return 0.99 if features.get('XBANK_flagged_in_registry') else 0.05

    bus = InMemoryEventBus()
    consumer = TransactionConsumer(
        feature_store=InMemoryFeatureStore(), scorer=scorer,
        cross_bank_adapter=MockCrossBankAdapter(seed=7),
    )
    bus.subscribe('transactions', consumer.handle_transaction)

    flagged_result = None
    for i in range(500):
        bus.publish('transactions', {
            'account_id': f'MOCK{i:04d}', 'timestamp': datetime.now(timezone.utc).isoformat(),
            'channel': 'CASH', 'direction': 'CREDIT', 'amount': 1000,
        })
        if consumer.event_log[-1].get('cross_bank_flagged'):
            flagged_result = consumer.event_log[-1]
            break

    assert flagged_result is not None, "expected at least one flagged account in 500 tries"
    assert flagged_result['action'] == 'AUTO-HOLD'
    assert flagged_result['score'] == 0.99


def test_auth_role_permission_matrix_admin_can_view_own_kill_switch():
    """Regression test for the exact bug found manually during Phase 5
    build: ADMIN could activate/deactivate the kill switch but was missing
    permission to view its own status. Locks that fix in place."""
    assert 'view_kill_switch_history' in ROLE_PERMISSIONS['ADMIN']
    assert 'kill_switch_activate' in ROLE_PERMISSIONS['ADMIN']


def test_auth_investigator_cannot_touch_kill_switch():
    assert 'kill_switch_activate' not in ROLE_PERMISSIONS['INVESTIGATOR']
    assert 'kill_switch_deactivate' not in ROLE_PERMISSIONS['INVESTIGATOR']


def test_auth_invalid_key_rejected():
    with pytest.raises(AuthorizationError):
        authenticate('not-a-real-key')


def test_api_requires_auth_header():
    client = app.test_client()
    r = client.get('/api/kill-switch/status')
    assert r.status_code == 401


def test_api_enforces_rbac_end_to_end():
    investigator_key = issue_api_key('inv_1', 'INVESTIGATOR')
    admin_key = issue_api_key('admin_1', 'ADMIN')
    client = app.test_client()

    # investigator forbidden from activating kill switch
    r = client.post('/api/kill-switch/activate', json={'reason': 'x'},
                    headers={'X-API-Key': investigator_key})
    assert r.status_code == 403

    # admin allowed
    r = client.post('/api/kill-switch/activate', json={'reason': 'x'},
                    headers={'X-API-Key': admin_key})
    assert r.status_code == 200

    client.post('/api/kill-switch/deactivate', json={'reason': 'cleanup'},
               headers={'X-API-Key': admin_key})


def test_api_health_needs_no_auth():
    client = app.test_client()
    assert client.get('/health').status_code == 200
