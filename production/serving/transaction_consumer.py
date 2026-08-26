"""
Transaction ingestion + scoring service — wires Phases 1, 2, and 3 together.

Pipeline: EventBus (Phase 3) delivers a transaction event -> FeatureStore
(Phase 2) is updated -> a pluggable scorer produces a risk score -> if the
score crosses the CRITICAL threshold, the governance layer (Phase 1) takes
over: kill_switch is checked first, then human_override.record_action()
logs the automated action, and customer_disclosure.generate_disclosure_notice()
prepares the customer-facing notice.

Honest scope note on the scorer: SACHET's trained ensemble
(models/ensemble.pkl) expects the exact ~150-column engineered feature
schema produced by src/satark.py's extract_features() pipeline, fit on the
competition dataset's real F-columns. The rolling features this module's
FeatureStore computes (feature_pipeline/feature_store.py) use a
channel-name schema derived from the mock/Finacle adapter, not that exact
column set. Force-feeding mismatched features into the real trained model
would produce meaningless scores, not a real integration — so `scorer` is
a pluggable callable here, not the real ensemble directly. The remaining
Phase-2 work is a schema-mapping layer that transforms live rolling
features into the trained model's exact expected columns; until that
mapping exists, this module demonstrates and tests the PIPELINE WIRING
(event -> feature update -> scoring hook -> governance action), not real
mule scores.

Cross-bank signal (optional): if a cross_bank_adapter is supplied (see
feature_pipeline/cross_bank_adapter.py), its XBANK_* features are merged
into the account's feature row before scoring. This is the concrete
architectural answer to the "genuinely undetectable" mules identified
earlier — accounts whose single-bank transaction stats look ordinary, but
whose counterparties are flagged elsewhere (via DPIP or I4C's Suspect
Registry once BOI has real access to either). With MockCrossBankAdapter
this is wiring/demo only, same honesty caveat as the scorer above.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'feature_pipeline'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'governance'))

from feature_store import compute_rolling_features, FeatureStore
from finacle_adapter import TransactionEvent
from cross_bank_adapter import CrossBankAdapter, signal_to_features
import kill_switch
import human_override
import customer_disclosure

CRITICAL_THRESHOLD = 0.95


class TransactionConsumer:
    def __init__(self, feature_store: FeatureStore, scorer, event_log: list = None,
                cross_bank_adapter: CrossBankAdapter = None):
        """scorer: callable(features: dict) -> float in [0, 1]. Swap in the
        real trained ensemble once the schema-mapping layer (see module
        docstring) exists; for now this accepts any callable so the pipeline
        wiring can be built and tested independently of that gap.
        cross_bank_adapter: optional CrossBankAdapter (DPIP/I4C) — if
        provided, its signal is merged into the feature row before scoring."""
        self.feature_store = feature_store
        self.scorer = scorer
        self.cross_bank_adapter = cross_bank_adapter
        self.event_log = event_log if event_log is not None else []
        self._recent_events_by_account = {}

    def handle_transaction(self, message: dict) -> dict:
        """The handler registered against the EventBus's 'transactions'
        topic. Returns a summary dict for logging/testing — real deployment
        would not need the return value, the side effects (feature update,
        governance action) are what matter."""
        event = TransactionEvent(
            account_id=message['account_id'],
            timestamp=message['timestamp'] if not isinstance(message['timestamp'], str)
                       else __import__('datetime').datetime.fromisoformat(message['timestamp']),
            channel=message['channel'],
            direction=message['direction'],
            amount=message['amount'],
        )
        self._recent_events_by_account.setdefault(event.account_id, []).append(event)

        features = compute_rolling_features(
            event.account_id, self._recent_events_by_account[event.account_id]
        )

        cross_bank_signal = None
        if self.cross_bank_adapter is not None:
            cross_bank_signal = self.cross_bank_adapter.fetch_signal(event.account_id)
            features.update(signal_to_features(cross_bank_signal))

        self.feature_store.update_online_features(event.account_id, features)

        score = self.scorer(features)
        result = {'account_id': event.account_id, 'score': score, 'action': None}
        if cross_bank_signal is not None:
            result['cross_bank_flagged'] = cross_bank_signal.flagged_in_suspect_registry

        if score >= CRITICAL_THRESHOLD:
            if kill_switch.is_active():
                result['action'] = 'SUPPRESSED_BY_KILL_SWITCH'
            else:
                action_id = human_override.record_action(
                    account_id=int(event.account_id.replace('MOCK', '') or 0)
                        if event.account_id.startswith('MOCK') else hash(event.account_id) % (10**8),
                    action_type='AUTO-HOLD',
                    action_detail=f'transaction score {score:.4f} on {event.channel}',
                    model_version='pipeline-wiring-demo',
                )
                notice = customer_disclosure.generate_disclosure_notice(
                    account_id=event.account_id, action_type='AUTO-HOLD',
                    reference_id=f'SACHET-{event.account_id}-{action_id}'
                )
                result['action'] = 'AUTO-HOLD'
                result['override_action_id'] = action_id
                result['customer_notice'] = notice['message']

        self.event_log.append(result)
        return result
