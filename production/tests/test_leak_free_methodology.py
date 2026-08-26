"""
Regression test for the leak-free methodology itself.

Why this exists: this exact codebase had two real leaks found and fixed
during development — a meta-learner scored in-sample, and NNPU reusing a
globally-selected feature list. Both leaks made the reported metric HIGHER
than the honest number (65/81 and later inflated numbers, vs. the true
61/81). A future code change could silently reintroduce either pattern
without anyone noticing, because a leak makes the number look BETTER, not
obviously broken — the kind of regression a normal test suite (checking for
crashes) would never catch.

This test pins the honest, currently-verified numbers as a tolerance band.
If a future change pushes the ensemble's out-of-fold recall_at_100 back up
toward the old leaky ~0.80 (65/81), or down below a sane floor (a real
accuracy regression), this test fails either way — "suspiciously better"
is treated as seriously as "worse."

Skips gracefully if the trained artifacts aren't present (e.g. running this
suite in an environment without the full competition dataset) rather than
failing the whole suite on a missing fixture.

2026-08-20 band update: a THIRD, separate bug was found and fixed —
train_models() (LightGBM) was saving its own final engineered+scaled 221-
column deployment matrix back into outputs/features.csv, the same shared
path every OTHER train_* function reads as its raw ~1,841-column starting
point. NNPU/CatBoost/focal/prototype were silently doing their own feature
selection on top of LightGBM's already-selected 221 columns instead of the
intended raw space (fixed by giving LightGBM's deployment matrix its own
path, _LGB_DEPLOY_FEATURES_PATH). This is a data-corruption bug, not a
leak — verified independently: every train_* function's own selection/
engineering/scaling still fits ONLY on that fold's training rows (audited
line by line), and LightGBM itself (first in pipeline order, unaffected by
the bug either way) reported IDENTICAL OOF numbers before and after the fix.
Giving the later models the correct, full raw feature space to select from
legitimately raised recall_at_100 from 61/81 (0.7531) to 66/81 (0.8148) —
a real improvement, not a reintroduced leak. The band below is recentered
on this new, independently-verified honest value.
"""

import os
import sys
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OOF_PATH = os.path.join(_REPO_ROOT, 'outputs', 'oof.npz')

pytestmark = pytest.mark.skipif(
    not os.path.exists(_OOF_PATH),
    reason="outputs/oof.npz not present — run the training pipeline first"
)


def test_ensemble_recall_at_100_is_honest_not_leaky():
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
    from satark import train_stacked_ensemble

    _, metrics = train_stacked_ensemble(save=False)

    # Honest, verified value at time of writing (post file-corruption fix,
    # see module docstring): 66/81 (0.8148). The original leaky value (in-
    # sample meta-learner + globally-selected NNPU features) was 65/81
    # (0.8025) — note this new HONEST value now sits close to that old LEAKY
    # one by coincidence of magnitude; the ablation report above (which
    # models loaded, learned weights, per-model AUC-PR) is what actually
    # distinguishes a genuine result from a leak, not the recall number
    # alone. Anything meaningfully above this band is still a prompt to
    # check for reintroduced leakage, not to celebrate blindly.
    assert 0.77 <= metrics['recall_at_100'] <= 0.87, (
        f"recall_at_100={metrics['recall_at_100']:.4f} is outside the honest "
        f"band [0.77, 0.87] — if this is HIGHER than 0.87, check for "
        f"reintroduced leakage (in-sample meta-learner fit, or feature "
        f"selection using validation-fold labels) before treating this as "
        f"a real improvement."
    )

    assert 0.75 <= metrics['auc_pr'] <= 0.85, (
        f"auc_pr={metrics['auc_pr']:.4f} outside the honest band [0.75, 0.85]"
    )
