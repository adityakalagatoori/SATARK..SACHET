"""
Full-ensemble scoring path for a NEW dataset (e.g. the hackathon's held-out
final-round file), as opposed to src/satark.py's predict_mules(), which only
runs the single LightGBM (Model B) booster.

This module reproduces the FULL pipeline the project actually reports numbers
for: 5 supervised/tree models (LightGBM, NNPU, CatBoost, Focal-loss LightGBM,
Prototype few-shot) + 2 unsupervised anomaly detectors (ECOD, DIF, shipped but
excluded from the ranking stack per DEFAULT_STACK_EXCLUDE) + the logistic-
regression meta-learner + the Stage-2 cascade precision-refiner + the
whole-population K-Means triage clusters + per-account SHAP explanations —
all using ONLY artifacts already fit and bundled in models/ensemble.pkl, or
deterministically reconstructable from the ORIGINAL training data already
committed in this repo (data/DataSet (1).csv / outputs/features.csv). Nothing
here ever calls .fit()/.fit_transform() on the newly uploaded data.

HONEST GAPS FOUND WHILE BUILDING THIS (see also the docstrings below):

1. Only Model B (LightGBM)'s deployment RobustScaler was persisted
   (models/scaler_v2.pkl -> bundle['scaler']) and only its top-150 feature
   list was persisted (models/top_features_v2.json -> bundle['top_features']).
   NNPU and CatBoost's train_* functions independently re-ran the identical
   mutual_info_classif(random_state=42) selection on the identical full
   dataset -> they provably land on the exact same 150 raw columns in the
   same order (verified: nnpu.feature_name() == lgb.feature_name() ==
   catboost.feature_names_ == bundle['top_features']-derived columns), so
   bundle['scaler'] is mathematically the correct scaler for them too. No
   approximation needed there.

2. The Focal-loss model (Model F) used BOOTSTRAP STABILITY SELECTION, a
   genuinely different (and non-reproducible-from-scratch, since it draws
   random bootstrap subsamples) feature list and therefore a different
   RobustScaler. Its own deployment feature list (models/focal_features_v2.json)
   was written by train_focal_loss_model() but then DELETED by
   bundle_models()'s cleanup_intermediates step as "debug-only" -- it was
   never carried into ensemble.pkl. However, LightGBM Boosters retain their
   OWN trained feature names internally (booster.feature_name()), so the
   exact 141-column list Focal was deployed on is still recoverable straight
   from the booster object with no guessing. This module recovers it that
   way, then reconstructs Focal's RobustScaler by fitting fresh on the
   ORIGINAL data/DataSet (1).csv (not the new upload) using that recovered
   feature list -- this reproduces the original artifact byte-for-byte
   (RobustScaler's median/IQR are a deterministic function of fixed input
   data), it is not "refitting on new data". The new/uploaded rows only ever
   go through `.transform()`.

3. whole_population_triage_clusters() fits a QuantileTransformer "fresh" on
   outputs/features.csv every run and never saves it (only the resulting
   KMeans is saved, as bundle['triage_kmeans']). Same treatment as (2): we
   reconstruct the QuantileTransformer by fitting it on the ORIGINAL
   outputs/features.csv (fixed, already-committed data) and only ever
   `.transform()` the new data before calling triage_kmeans.predict()
   (never refit KMeans itself).

4. ECOD/DIF normalize their decision_function() output to [0,1] using the
   min/max of the SET BEING SCORED, which was never saved either. We fix this
   by computing the min/max ONE TIME on the original training population
   (a pure inference call, not a fit) and freezing it as the normalization
   reference for every subsequent scoring call -- otherwise every new batch
   would rescale to its own private, non-comparable [0,1] range.

5. If the uploaded CSV's F-columns don't match the schema this bundle was
   trained on, we raise ValueError with the exact column diff instead of
   silently reindexing -- see `SchemaMismatchError`.
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

import satark  # noqa: E402  (repo's src/ package)

_BUNDLE_PATH = os.path.join(_REPO_ROOT, 'models', 'ensemble.pkl')
_ORIGINAL_DATA_PATH = os.path.join(_REPO_ROOT, 'data', 'DataSet (1).csv')
_REFERENCE_FEATURES_PATH = os.path.join(_REPO_ROOT, 'outputs', 'features.csv')

_LABEL_COL = 'F3924'
_STACK_MODEL_ORDER = ['nnpu', 'lgb', 'catboost', 'focal']  # excludes ecod/dif/proto/tabpfn, matches DEFAULT_STACK_EXCLUDE


class SchemaMismatchError(ValueError):
    pass


# ============================================================================
# CACHED, ONE-TIME-RECONSTRUCTED ARTIFACTS
# ============================================================================

_cache = {}


def _load_bundle():
    if 'bundle' not in _cache:
        if not os.path.exists(_BUNDLE_PATH):
            raise FileNotFoundError(f"models/ensemble.pkl not found at {_BUNDLE_PATH}")
        _cache['bundle'] = joblib.load(_BUNDLE_PATH)
    return _cache['bundle']


def _load_reference_raw():
    """The ORIGINAL cleaned (label-independent) feature matrix this bundle
    was trained/deployed on -- outputs/features.csv. Used only as: (a) the
    canonical column schema new data must be reindexed to, (b) fixed fit data
    to deterministically reconstruct the artifacts noted as gaps above. Never
    mutated, never refit on incoming data."""
    if 'ref_raw' not in _cache:
        if not os.path.exists(_REFERENCE_FEATURES_PATH):
            raise FileNotFoundError(
                f"{_REFERENCE_FEATURES_PATH} not found -- run satark.py --mode extract first "
                f"(needed as the fixed reference schema/fit-data for this scoring path)."
            )
        _cache['ref_raw'] = pd.read_csv(_REFERENCE_FEATURES_PATH)
    return _cache['ref_raw']


def _load_original_labels():
    if 'y_original' not in _cache:
        df = pd.read_csv(_ORIGINAL_DATA_PATH)
        _cache['y_original'] = df[_LABEL_COL].values.astype(int) if _LABEL_COL in df.columns else None
    return _cache['y_original']


# ---------------------------------------------------------------------------
# Raw feature reconstruction (encode_categoricals + typology features +
# numeric filter + reindex-to-reference-schema). This "applies" the ALREADY
# DECIDED constant-removal/correlation-pruning column set from
# outputs/features.csv rather than re-deriving it from the new data (which
# would be a forbidden refit of feature selection).
# ---------------------------------------------------------------------------

def build_reference_matched_raw_features(raw_df: pd.DataFrame, strict: bool = True):
    """Rebuild the label-independent engineered-but-unselected matrix for new
    data, reindexed onto the EXACT column schema of outputs/features.csv (the
    schema every downstream model expects).

    Returns (X_raw, report) where report documents any schema drift found.
    Raises SchemaMismatchError if strict and the new data is missing so many
    of the required raw source columns that reconstruction cannot proceed
    honestly (rather than silently filling everything with medians).
    """
    ref_raw = _load_reference_raw()
    ref_cols = list(ref_raw.columns)

    df = raw_df.copy()
    leak_cols = [c for c in [_LABEL_COL, 'Unnamed: 0', 'F2230',
                              'F3912', 'F3913', 'F3914', 'F3915'] if c in df.columns]
    X = df.drop(columns=leak_cols)

    X_cat = satark.encode_categoricals(X)
    X_typ = satark.add_typology_features(X, return_only_new=True)

    X_num = X.select_dtypes(include=[np.number])
    X_all = pd.concat([X_num, X_cat, X_typ], axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()]

    present = [c for c in ref_cols if c in X_all.columns]
    missing = [c for c in ref_cols if c not in X_all.columns]
    extra = [c for c in X_all.columns if c not in ref_cols]

    missing_frac = len(missing) / max(len(ref_cols), 1)
    if strict and missing_frac > 0.5:
        raise SchemaMismatchError(
            f"Uploaded dataset is missing {len(missing)}/{len(ref_cols)} "
            f"({missing_frac*100:.1f}%) of the raw engineered columns this model bundle "
            f"expects (derived from the original 3,924 F-columns schema). This looks like "
            f"a genuine schema mismatch, not just a handful of dropped columns -- refusing "
            f"to silently impute the majority of the feature space. First 20 missing: "
            f"{missing[:20]}"
        )

    X_ref = pd.DataFrame(index=X_all.index, columns=ref_cols, dtype=float)
    for c in present:
        X_ref[c] = pd.to_numeric(X_all[c], errors='coerce')
    # Missing required columns: impute with the ORIGINAL reference population's
    # median for that column (a fixed, already-observed statistic) -- never a
    # median computed from the new upload.
    ref_medians = ref_raw.median(numeric_only=True)
    for c in missing:
        X_ref[c] = ref_medians.get(c, 0.0)
    # Any NaNs remaining within columns that WERE present (e.g. new-data-only
    # missing values) also get the reference median, not a new-data median.
    for c in present:
        if X_ref[c].isnull().any():
            X_ref[c] = X_ref[c].fillna(ref_medians.get(c, 0.0))

    report = {
        'raw_columns_expected': len(ref_cols),
        'raw_columns_matched': len(present),
        'raw_columns_missing_imputed': missing,
        'raw_columns_extra_dropped': extra,
        'missing_fraction': missing_frac,
    }
    return X_ref, report


# ---------------------------------------------------------------------------
# Per-model engineered-feature reconstruction, using each booster/model's OWN
# feature_name() list to recover exactly which top-N raw columns (and in what
# order) + which log-transformed columns it was deployed on.
# ---------------------------------------------------------------------------

def _recover_top_features_and_logcols(feature_names, raw_col_set):
    """feature_name() lists are built by _engineer_from_top_features() as:
    [top_features..., <some>_log..., <top_features[:20]>_sq..., <interactions
    among top_features[:10]>..., 'top_10_mean','top_10_std','top_20_mean',
    'top_20_std','top_50_mean']. The raw top_features block always comes
    first and consists exactly of names that are themselves raw column names,
    so a simple prefix scan recovers it losslessly."""
    top_features = []
    for name in feature_names:
        if name in raw_col_set:
            top_features.append(name)
        else:
            break
    log_cols = [name[:-4] for name in feature_names
                if name.endswith('_log') and name[:-4] in top_features]
    return top_features, log_cols


def _engineered_matrix_for_model(X_ref_raw: pd.DataFrame, feature_names, raw_col_set):
    top_features, log_cols = _recover_top_features_and_logcols(feature_names, raw_col_set)
    eng = satark._engineer_from_top_features(X_ref_raw, top_features, log_cols=log_cols)
    missing_engineered = [c for c in feature_names if c not in eng.columns]
    if missing_engineered:
        raise RuntimeError(
            f"Feature reconstruction did not reproduce {len(missing_engineered)} columns "
            f"the model expects (e.g. {missing_engineered[:5]}) -- aborting rather than "
            f"silently scoring on a misaligned matrix."
        )
    return eng[feature_names]  # exact order the model was trained on


def _focal_scaler():
    """Reconstruct Focal's own RobustScaler by refitting on the ORIGINAL
    (already-committed) reference data using its recovered feature list --
    see module docstring gap #2. Cached process-wide; only ever fit once,
    on fixed original data, never on incoming uploads."""
    if 'focal_scaler' not in _cache:
        bundle = _load_bundle()
        focal_model = bundle['focal']
        ref_raw = _load_reference_raw()
        raw_col_set = set(ref_raw.columns)
        feature_names = focal_model.feature_name()
        top_features, log_cols = _recover_top_features_and_logcols(feature_names, raw_col_set)
        eng_ref = satark._engineer_from_top_features(ref_raw, top_features, log_cols=log_cols)[feature_names]
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        scaler.fit(eng_ref)
        _cache['focal_scaler'] = scaler
        _cache['focal_feature_names'] = feature_names
    return _cache['focal_scaler'], _cache['focal_feature_names']


def _quantile_transformer_for_triage():
    """Reconstruct whole_population_triage_clusters()'s QuantileTransformer
    -- see module docstring gap #3. Fit once on the fixed original
    outputs/features.csv, cached; new data only ever goes through
    .transform()."""
    if 'triage_qt' not in _cache:
        from sklearn.preprocessing import QuantileTransformer
        ref_raw = _load_reference_raw()
        qt = QuantileTransformer(output_distribution='uniform', random_state=42)
        qt.fit(ref_raw)
        _cache['triage_qt'] = qt
    return _cache['triage_qt']


def _universe_pca():
    """3D PCA projection consistent with presentation/account_universe_coords.json's
    coordinate space, so newly uploaded accounts land in the SAME space as the
    existing 3D universe (project new points in, never re-fit a new space).

    account_universe_coords.json itself only stored the resulting x/y/z, not
    the fitted PCA object -- another artifact that was never persisted. We
    reconstruct it by fitting PCA(n_components=3) once on the ORIGINAL
    dataset's QuantileTransformer-rank-transformed raw feature space (the
    SAME space whole_population_triage_clusters() already uses for K-Means,
    for the same reason: the engineered matrix's squared/interaction columns
    have extreme, outlier-dominated magnitudes that make raw/RobustScaler-
    scaled PCA coordinates explode into the tens of thousands and not be
    comparable to the existing coords file's ~[-1,1] range). New accounts are
    only ever .transform()'d through both the QuantileTransformer and PCA,
    never refit."""
    if 'universe_pca' not in _cache:
        from sklearn.decomposition import PCA
        ref_raw = _load_reference_raw()
        qt = _quantile_transformer_for_triage()
        X_ref_q = qt.transform(ref_raw)
        pca = PCA(n_components=3, random_state=42)
        pca.fit(X_ref_q)
        _cache['universe_pca'] = pca
    return _cache['universe_pca']


def _anomaly_score_ranges():
    """Freeze ECOD/DIF's normalization min/max against the ORIGINAL training
    population -- see module docstring gap #4. Calls to .decision_function()
    here are pure inference (the models are already fit), not training."""
    if 'anomaly_ranges' not in _cache:
        bundle = _load_bundle()
        ref_raw = _load_reference_raw()
        ranges = {}
        for key in ('ecod', 'dif'):
            if key in bundle:
                scores = bundle[key].decision_function(ref_raw)
                ranges[key] = (float(scores.min()), float(scores.max()))
        _cache['anomaly_ranges'] = ranges
    return _cache['anomaly_ranges']


# ============================================================================
# MAIN SCORING ENTRY POINT
# ============================================================================

def score_dataset(raw_df: pd.DataFrame, top_n_shap: int = 10, compute_shap: bool = True,
                   strict_schema: bool = True):
    """Score every row of `raw_df` (same 3,924 F-column raw schema as
    data/DataSet (1).csv, minus F3924 if unlabeled) through the FULL bundled
    ensemble: 5 supervised models, ECOD/DIF (shipped, not stacked), the
    logistic-regression meta-learner, the Stage-2 cascade, K-Means triage
    cluster, and per-account SHAP (if available).

    Returns a dict:
      {
        'meta': {...run-level info, including any labels-derived metrics if
                 F3924 was present, and the schema report...},
        'accounts': [ {id, base:{lgb,nnpu,catboost,focal}, anomaly:{ecod,dif},
                       proto, main (meta-learner), cascade, cluster, shap,
                       is_mule (if labels present)} , ... ]
      }
    """
    bundle = _load_bundle()

    has_labels = _LABEL_COL in raw_df.columns
    y_true = raw_df[_LABEL_COL].values.astype(int) if has_labels else None

    X_ref_raw, schema_report = build_reference_matched_raw_features(raw_df, strict=strict_schema)
    n = len(X_ref_raw)
    raw_col_set = set(X_ref_raw.columns)

    # ---- Model B: LightGBM (scaler + top_features ARE saved in the bundle) ----
    lgb_feat_names = bundle['lgb'].feature_name()
    X_lgb = _engineered_matrix_for_model(X_ref_raw, lgb_feat_names, raw_col_set)
    X_lgb_scaled = pd.DataFrame(bundle['scaler'].transform(X_lgb), columns=X_lgb.columns, index=X_lgb.index)
    lgb_scores = bundle['lgb'].predict(X_lgb_scaled)

    # ---- Model A: NNPU (identical top_features/log_cols/scaler to lgb -- verified) ----
    nnpu_feat_names = bundle['nnpu'].feature_name()
    if nnpu_feat_names == lgb_feat_names:
        X_nnpu_scaled = X_lgb_scaled
    else:
        X_nnpu = _engineered_matrix_for_model(X_ref_raw, nnpu_feat_names, raw_col_set)
        X_nnpu_scaled = pd.DataFrame(bundle['scaler'].transform(X_nnpu), columns=X_nnpu.columns, index=X_nnpu.index)
    nnpu_raw = bundle['nnpu'].predict(X_nnpu_scaled, raw_score=True)
    nnpu_scores = 1.0 / (1.0 + np.exp(-nnpu_raw))

    # ---- Model E: CatBoost (same as NNPU) ----
    cb_feat_names = list(bundle['catboost'].feature_names_)
    if cb_feat_names == lgb_feat_names:
        X_cb_scaled = X_lgb_scaled
    else:
        X_cb = _engineered_matrix_for_model(X_ref_raw, cb_feat_names, raw_col_set)
        X_cb_scaled = pd.DataFrame(bundle['scaler'].transform(X_cb), columns=X_cb.columns, index=X_cb.index)
    catboost_scores = bundle['catboost'].predict_proba(X_cb_scaled)[:, 1]

    # ---- Model F: Focal-loss LightGBM (its own recovered feature list + reconstructed scaler) ----
    focal_scaler, focal_feat_names = _focal_scaler()
    X_focal = _engineered_matrix_for_model(X_ref_raw, focal_feat_names, raw_col_set)
    X_focal_scaled = pd.DataFrame(focal_scaler.transform(X_focal), columns=X_focal.columns, index=X_focal.index)
    focal_raw = bundle['focal'].predict(X_focal_scaled, raw_score=True)
    focal_scores = 1.0 / (1.0 + np.exp(-focal_raw))

    # ---- Model G: Prototype few-shot (fully self-contained in bundle['proto']) ----
    proto = bundle.get('proto')
    if proto is not None:
        # proto['scaler'].feature_names_in_ is the actual fully-engineered
        # column list (raw + log/sq/interaction/aggregate) it was fit on --
        # using that (rather than re-deriving log_cols fresh from whatever
        # rows are being scored) is required for the same reason boosters'
        # feature_name() is used above: log_cols must match the ORIGINAL
        # fit-time decision, not be recomputed per batch.
        proto_scaler_feat_names = list(proto['scaler'].feature_names_in_)
        proto_eng = _engineered_matrix_for_model(X_ref_raw, proto_scaler_feat_names, raw_col_set)
        proto_scaled = proto['scaler'].transform(proto_eng)
        pos_c, neg_c = proto['pos_centers'], proto['neg_centers']
        d_pos = np.min(np.linalg.norm(proto_scaled[:, None, :] - pos_c[None, :, :], axis=2), axis=1)
        d_neg = np.min(np.linalg.norm(proto_scaled[:, None, :] - neg_c[None, :, :], axis=2), axis=1)
        proto_scores = d_neg / (d_pos + d_neg + 1e-9)
    else:
        proto_scores = np.full(n, np.nan)

    # ---- Model C: ECOD / DIF (unsupervised, shipped but excluded from the stack) ----
    anomaly_ranges = _anomaly_score_ranges()
    anomaly_scores = {}
    for key in ('ecod', 'dif'):
        if key in bundle:
            raw_scores = bundle[key].decision_function(X_ref_raw)
            lo, hi = anomaly_ranges[key]
            anomaly_scores[key] = (raw_scores - lo) / (hi - lo + 1e-10)
            anomaly_scores[key] = np.clip(anomaly_scores[key], 0.0, 1.0)

    # ---- Meta-learner: stack nnpu/lgb/catboost/focal (DEFAULT_STACK_EXCLUDE = ecod, dif) ----
    meta_learner = bundle['meta_learner']
    stack_cols = list(getattr(meta_learner, 'feature_names_in_', _STACK_MODEL_ORDER))
    base_scores = {'nnpu': nnpu_scores, 'lgb': lgb_scores, 'catboost': catboost_scores, 'focal': focal_scores,
                   'proto': proto_scores, **anomaly_scores}
    meta_X = pd.DataFrame({c: base_scores[c] for c in stack_cols})
    main_scores = meta_learner.predict_proba(meta_X)[:, 1]

    # ---- Stage-2 cascade precision-refiner ----
    cascade = bundle.get('cascade')
    if cascade is not None:
        cascade_feats = cascade['features']
        missing_cascade = [c for c in cascade_feats if c not in X_ref_raw.columns]
        if missing_cascade:
            raise RuntimeError(f"Cascade expects raw columns not present after reconstruction: {missing_cascade[:5]}")
        cascade_scores = cascade['model'].predict(X_ref_raw[cascade_feats])
    else:
        cascade_scores = np.full(n, np.nan)

    # ---- Whole-population triage cluster assignment ----
    triage_kmeans = bundle.get('triage_kmeans')
    if triage_kmeans is not None:
        qt = _quantile_transformer_for_triage()
        X_for_cluster = qt.transform(X_ref_raw[list(_load_reference_raw().columns)])
        cluster_ids = triage_kmeans.predict(X_for_cluster)
    else:
        cluster_ids = np.full(n, -1)

    # ---- SHAP (reuses lgb's TreeExplainer, same approach as explain_prediction()) ----
    shap_per_row = [None] * n
    shap_error = None
    if compute_shap:
        try:
            import shap
            explainer = shap.TreeExplainer(bundle['lgb'])
            shap_values = explainer.shap_values(X_lgb_scaled)
            vals = shap_values[0] if isinstance(shap_values, list) else shap_values
            vals = np.asarray(vals)
            cols = X_lgb_scaled.columns
            for i in range(n):
                row_vals = vals[i]
                pairs = sorted(zip(cols, row_vals), key=lambda kv: abs(kv[1]), reverse=True)[:top_n_shap]
                shap_per_row[i] = [
                    {'feature': str(f), 'shap_value': float(v), 'abs_value': float(abs(v)),
                     'direction': 'increases' if v > 0 else 'decreases'}
                    for f, v in pairs
                ]
        except ImportError:
            shap_error = 'shap not installed'
        except Exception as e:  # pragma: no cover - defensive
            shap_error = str(e)

    # ---- 3D coordinates: project into the SAME PCA space as the original
    # account universe (presentation/account_universe_coords.json) -- see
    # _universe_pca() docstring for why this must be .transform() only. ----
    pca = _universe_pca()
    qt = _quantile_transformer_for_triage()
    coords = pca.transform(qt.transform(X_ref_raw[list(_load_reference_raw().columns)]))[:, :3]

    # ---- severity / recommended action, same thresholds as
    # src/satark.py's generate_alerts() ----
    def _severity(score100):
        if score100 >= 95:
            return 'CRITICAL', ('AUTO-HOLD: temporary step-up verification on large outgoing '
                                 'transfers (reversible, not a freeze) — pending same-day review')
        if score100 >= 85:
            return 'HIGH', 'PRIORITY MANUAL REVIEW — same-day queue'
        if score100 >= 70:
            return 'MEDIUM', 'STANDARD QUEUE — review within the week'
        if score100 >= 50:
            return 'LOW', 'WATCHLIST — periodic batch review, no immediate action'
        return 'NONE', 'No alert'

    accounts = []
    tier_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'NONE': 0}
    for i in range(n):
        main100 = float(main_scores[i]) * 100
        severity, action = _severity(main100)
        tier_counts[severity] += 1
        rec = {
            'id': int(i),
            'base': {
                'lgb': round(float(lgb_scores[i]) * 100, 4),
                'nnpu': round(float(nnpu_scores[i]) * 100, 4),
                'catboost': round(float(catboost_scores[i]) * 100, 4),
                'focal': round(float(focal_scores[i]) * 100, 4),
            },
            'anomaly': {k: round(float(v[i]) * 100, 4) for k, v in anomaly_scores.items()},
            'proto': None if np.isnan(proto_scores[i]) else round(float(proto_scores[i]) * 100, 4),
            'main': round(main100, 4),
            'score': round(main100, 4),
            'cascade': None if np.isnan(cascade_scores[i]) else round(float(cascade_scores[i]) * 100, 4),
            'cluster': int(cluster_ids[i]),
            'severity': severity,
            'action': action,
            'x': round(float(coords[i, 0]), 4),
            'y': round(float(coords[i, 1]), 4),
            'z': round(float(coords[i, 2]), 4),
            'risk_score': round(main100 / 100.0, 4),
            'shap': shap_per_row[i],
        }
        if has_labels:
            rec['is_mule'] = int(y_true[i])
        accounts.append(rec)

    meta_info = {
        'n_accounts': n,
        'has_labels': has_labels,
        'schema_report': schema_report,
        'shap_error': shap_error,
        'tier_counts': tier_counts,
        'gaps': {
            'focal_scaler': 'reconstructed from original training data (not persisted in bundle)',
            'triage_quantile_transformer': 'reconstructed from original outputs/features.csv (not persisted in bundle)',
            'ecod_dif_normalization_range': 'frozen from original training population decision_function() output',
            'universe_pca': 'PCA(3) refit on the original lgb-deployment feature space (not persisted alongside '
                             'presentation/account_universe_coords.json); new accounts are only ever .transform()-ed '
                             'through it, so coordinates are consistent with, but not a byte-identical reproduction '
                             'of, the original coords file.',
        },
    }

    if has_labels:
        from sklearn.metrics import precision_recall_curve, auc as compute_auc, roc_auc_score
        p, r, _ = precision_recall_curve(y_true, main_scores)
        meta_info['metrics'] = {
            'main_ensemble_auc_pr': float(compute_auc(r, p)),
            'main_ensemble_roc_auc': float(roc_auc_score(y_true, main_scores)),
            'n_mules': int(y_true.sum()),
        }
        for name, scores in (('lgb', lgb_scores), ('nnpu', nnpu_scores),
                              ('catboost', catboost_scores), ('focal', focal_scores)):
            pp, rr, _ = precision_recall_curve(y_true, scores)
            meta_info['metrics'][f'{name}_auc_pr'] = float(compute_auc(rr, pp))
    else:
        meta_info['metrics'] = None
        meta_info['metrics_note'] = ('No F3924 ground-truth column found in the uploaded file -- '
                                      'AUC-PR/recall/precision cannot be computed and are not fabricated.')

    return {'meta': meta_info, 'accounts': accounts}


def score_csv_file(path: str, **kwargs):
    raw_df = pd.read_csv(path)
    return score_dataset(raw_df, **kwargs)


# ============================================================================
# UPLOAD PERSISTENCE — so a scored dataset can be selected later in the
# Investigator Console / opened per-account in the 3D view as a plain static
# JSON file (no backend needed to VIEW a result once it has been scored once;
# the backend is only needed at scoring time).
# ============================================================================

_UPLOADS_DIR = os.path.join(_REPO_ROOT, 'outputs', 'uploads')
_UPLOADS_INDEX_PATH = os.path.join(_UPLOADS_DIR, 'index.json')


def _load_uploads_index():
    if os.path.exists(_UPLOADS_INDEX_PATH):
        with open(_UPLOADS_INDEX_PATH) as f:
            return json.load(f)
    return []


def _save_uploads_index(index):
    os.makedirs(_UPLOADS_DIR, exist_ok=True)
    with open(_UPLOADS_INDEX_PATH, 'w') as f:
        json.dump(index, f, indent=2)


def score_and_persist(raw_df: pd.DataFrame, filename: str = 'upload.csv', compute_shap: bool = True):
    """Score `raw_df` through the full pipeline and persist the result as
    outputs/uploads/<upload_id>.json, registering it in
    outputs/uploads/index.json. Returns {'upload_id':..., 'summary': {...}}.

    Each upload gets its own id/file -- nothing here overwrites a previous
    upload's results."""
    from datetime import datetime
    now = datetime.now()
    upload_id = 'upload_' + now.strftime('%Y%m%d_%H%M%S')
    os.makedirs(_UPLOADS_DIR, exist_ok=True)
    # guard against two uploads landing in the same second
    suffix = 0
    base_id = upload_id
    while os.path.exists(os.path.join(_UPLOADS_DIR, f'{upload_id}.json')):
        suffix += 1
        upload_id = f'{base_id}_{suffix}'

    result = score_dataset(raw_df, compute_shap=compute_shap)
    result['upload_id'] = upload_id
    result['filename'] = filename
    result['scored_at'] = now.isoformat()

    with open(os.path.join(_UPLOADS_DIR, f'{upload_id}.json'), 'w') as f:
        json.dump(result, f)

    index = _load_uploads_index()
    summary = {
        'upload_id': upload_id,
        'filename': filename,
        'scored_at': now.isoformat(),
        'n_accounts': result['meta']['n_accounts'],
        'has_labels': result['meta']['has_labels'],
        'tier_counts': result['meta']['tier_counts'],
        'metrics': result['meta'].get('metrics'),
    }
    index.append(summary)
    _save_uploads_index(index)

    return {'upload_id': upload_id, 'summary': summary}


def load_upload(upload_id: str):
    path = os.path.join(_UPLOADS_DIR, f'{upload_id}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('csv_path', nargs='?', default=_ORIGINAL_DATA_PATH)
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-shap', action='store_true')
    args = ap.parse_args()

    result = score_csv_file(args.csv_path, compute_shap=not args.no_shap)
    print(json.dumps(result['meta'], indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f)
        print(f"Saved full result to {args.out}")
