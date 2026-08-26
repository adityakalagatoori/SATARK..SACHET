"""
Feature and prediction drift monitoring — Population Stability Index (PSI).

Why this exists: a model that was honestly validated at deployment time can
silently degrade as real-world account behavior shifts (new fraud typologies,
seasonal spending changes, product mix changes) — accuracy doesn't visibly
drop until it's already been wrong for a while. PSI is the standard
technique for catching this early: it compares the distribution of a feature
(or the model's own prediction scores) between a baseline period and a
current period, without needing new labels — which matters here, since
confirmed-mule labels only arrive after a slow investigation, long after
drift could have started.

PSI interpretation (industry-standard thresholds):
  < 0.1  — no significant change
  0.1-0.25 — moderate shift, worth investigating
  > 0.25 — significant shift, likely warrants retraining or a kill-switch review
"""

import numpy as np


def compute_psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """PSI between two 1D numeric arrays. Bins are defined on the baseline
    distribution's quantiles, then both distributions are compared against
    those same bin edges — this is why baseline must be the training-time
    distribution and current must be live data, not the other way round."""
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(baseline, quantiles))
    if len(bin_edges) < 2:
        return 0.0  # baseline has no variance in this feature — PSI undefined, treat as no drift signal

    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    baseline_pct = np.maximum(baseline_counts / len(baseline), 1e-6)
    current_pct = np.maximum(current_counts / len(current), 1e-6)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def psi_severity(psi: float) -> str:
    if psi < 0.1:
        return 'STABLE'
    elif psi < 0.25:
        return 'MODERATE_SHIFT'
    else:
        return 'SIGNIFICANT_SHIFT'


def monitor_features(baseline_df, current_df, feature_columns: list, n_bins: int = 10) -> dict:
    """Run PSI across a set of features (typically the model's top selected
    features) and return a per-feature report, sorted worst-first — the
    features to look at when deciding whether to trigger a retrain."""
    results = []
    for col in feature_columns:
        if col not in baseline_df.columns or col not in current_df.columns:
            continue
        psi = compute_psi(baseline_df[col].values, current_df[col].values, n_bins=n_bins)
        results.append({'feature': col, 'psi': psi, 'severity': psi_severity(psi)})

    results.sort(key=lambda r: r['psi'], reverse=True)
    n_significant = sum(1 for r in results if r['severity'] == 'SIGNIFICANT_SHIFT')
    return {
        'features': results,
        'n_significant_shift': n_significant,
        'overall_recommendation': (
            'RETRAIN_REVIEW_RECOMMENDED' if n_significant > 0 else 'NO_ACTION_NEEDED'
        ),
    }


def monitor_prediction_scores(baseline_scores, current_scores, n_bins: int = 10) -> dict:
    """PSI on the model's own output distribution — catches drift even when
    no single input feature moved enough to flag on its own, but the
    combination shifted the model's overall risk-score distribution."""
    psi = compute_psi(np.asarray(baseline_scores), np.asarray(current_scores), n_bins=n_bins)
    return {'psi': psi, 'severity': psi_severity(psi)}
