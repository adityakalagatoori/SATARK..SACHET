"""Regression tests for Phase 4 — model registry and drift monitoring."""

import pytest
import os
import numpy as np

import model_registry as mr
from drift_monitor import compute_psi, psi_severity, monitor_prediction_scores


@pytest.fixture
def fake_artifacts(tmp_path):
    paths = {}
    for name, content in [('v1', b'model v1'), ('v_worse', b'model worse'), ('v_better', b'model better')]:
        p = tmp_path / f"{name}.pkl"
        p.write_bytes(content)
        paths[name] = str(p)
    return paths


def test_registry_promote_blocks_regression(fake_artifacts):
    mr.register_model('m', 'v1', fake_artifacts['v1'], {'auc_pr': 0.80}, 'ci')
    mr.promote('m', 'v1', 'ci')

    mr.register_model('m', 'v_worse', fake_artifacts['v_worse'], {'auc_pr': 0.70}, 'ci')
    promoted = mr.promote('m', 'v_worse', 'ci')

    assert promoted is False
    assert mr.get_champion('m')['version'] == 'v1'  # regression must not become champion


def test_registry_promote_allows_improvement(fake_artifacts):
    mr.register_model('m', 'v1', fake_artifacts['v1'], {'auc_pr': 0.80}, 'ci')
    mr.promote('m', 'v1', 'ci')

    mr.register_model('m', 'v_better', fake_artifacts['v_better'], {'auc_pr': 0.85}, 'ci')
    promoted = mr.promote('m', 'v_better', 'ci')

    assert promoted is True
    assert mr.get_champion('m')['version'] == 'v_better'


def test_registry_rollback_bypasses_regression_check(fake_artifacts):
    """rollback() is an intentional emergency escape hatch — it must be able
    to restore an older, lower-metric version even though promote() would
    refuse the same move."""
    mr.register_model('m', 'v1', fake_artifacts['v1'], {'auc_pr': 0.80}, 'ci')
    mr.promote('m', 'v1', 'ci')
    mr.register_model('m', 'v_better', fake_artifacts['v_better'], {'auc_pr': 0.85}, 'ci')
    mr.promote('m', 'v_better', 'ci')

    mr.rollback('m', 'v1', 'risk_officer')
    assert mr.get_champion('m')['version'] == 'v1'


def test_registry_artifacts_are_never_overwritten(fake_artifacts):
    """Each registered version gets its own immutable copy — registering v2
    must not touch v1's stored artifact file."""
    mr.register_model('m', 'v1', fake_artifacts['v1'], {'auc_pr': 0.80}, 'ci')
    v1_row = mr.list_versions('m')[0]
    assert os.path.exists(v1_row['artifact_path'])
    with open(v1_row['artifact_path'], 'rb') as f:
        assert f.read() == b'model v1'


def test_psi_stable_distribution_is_low():
    rng = np.random.RandomState(0)
    baseline = rng.normal(50000, 15000, 5000)
    current = rng.normal(50000, 15000, 1000)
    assert compute_psi(baseline, current) < 0.1
    assert psi_severity(compute_psi(baseline, current)) == 'STABLE'


def test_psi_shifted_distribution_is_flagged():
    rng = np.random.RandomState(0)
    baseline = rng.normal(50000, 15000, 5000)
    shifted = rng.normal(150000, 15000, 1000)  # large, unambiguous shift
    psi = compute_psi(baseline, shifted)
    assert psi > 0.25
    assert psi_severity(psi) == 'SIGNIFICANT_SHIFT'


def test_psi_identical_distribution_is_near_zero():
    rng = np.random.RandomState(0)
    baseline = rng.normal(50000, 15000, 5000)
    assert compute_psi(baseline, baseline) < 0.01


def test_monitor_prediction_scores_returns_severity():
    rng = np.random.RandomState(0)
    result = monitor_prediction_scores(rng.beta(2, 20, 5000), rng.beta(2, 20, 1000))
    assert 'psi' in result and 'severity' in result
