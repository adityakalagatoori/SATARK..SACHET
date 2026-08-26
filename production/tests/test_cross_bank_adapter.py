"""Regression tests for the cross-bank (DPIP/I4C) signal adapter."""

import pytest
from cross_bank_adapter import MockCrossBankAdapter, DPIPAdapter, signal_to_features


def test_mock_adapter_returns_valid_signal():
    adapter = MockCrossBankAdapter(seed=1)
    sig = adapter.fetch_signal('ACC1')
    assert sig.account_id == 'ACC1'
    assert isinstance(sig.flagged_in_suspect_registry, bool)
    assert 0.0 <= sig.cross_bank_velocity_score <= 1.0
    assert sig.source == 'MOCK'


def test_signal_to_features_shape():
    adapter = MockCrossBankAdapter(seed=1)
    sig = adapter.fetch_signal('ACC1')
    feats = signal_to_features(sig)
    assert set(feats.keys()) == {
        'XBANK_flagged_in_registry', 'XBANK_counterparty_flagged_count', 'XBANK_velocity_score'
    }


def test_real_dpip_adapter_refuses_to_fake_success():
    """The DPIP adapter must never silently behave as if it has real access
    — it should fail loudly at construction, not return plausible-looking
    fake data that could be mistaken for a real integration."""
    with pytest.raises(NotImplementedError):
        DPIPAdapter(api_endpoint='https://dpip.rbi.org.in', api_credentials={})
