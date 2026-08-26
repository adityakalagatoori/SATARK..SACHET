# SACHET

**AI/ML Mule Account Detection System**
Team SATARK (सतर्क = "Vigilant") — PSB Cybersecurity, Fraud & AI Hackathon 2026, Problem Statement 2 (Bank of India × IIT Hyderabad)

SACHET flags suspicious mule accounts from bank transaction data using a stacked
ensemble (cost-sensitive LightGBM, Positive-Unlabeled learning, CatBoost, a
focal-loss LightGBM variant, and a prototype/metric-learning model), refined by
a second-stage cascade classifier, with leakage-free nested cross-validation
throughout. On the official competition dataset (9,082 accounts, 81 confirmed
mules, 0.89% prevalence), the ensemble achieves **0.8229 AUC-PR** (91×+ the
random baseline), **66% Precision@100**, and **66/81 (81.5%) Recall@100** —
every number is out-of-fold, never scored on rows used to select features,
scale, or train.

Unsupervised K-Means triage additionally concentrates **69 of 81 confirmed
mules (85.2%) into a single cluster** spanning under half the account
population — usable today as an investigator prioritization queue, without
needing the supervised model at all.

## Team

| Member | Focus | Institution |
|---|---|---|
| Kritika Pandey | AI/ML | United College of Engineering and Research, Prayagraj |
| Vaishnav Aditya | Cyber Security | Amrita Vishwa Vidyapeetham, Chennai |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the official dataset

Put the competition files in `data/`:
- `data/DataSet (1).csv`
- `data/Description.xlsx`

### 3. Run the full pipeline

```bash
# Feature engineering + all base models + cascade + meta-learner + triage clustering + bundling
python src/satark.py --mode full-ensemble
```

This runs, in order: `extract` → `train` (LightGBM) → `train-nnpu` → `anomaly`
(ECOD + Deep Isolation Forest) → `train-catboost` → `train-focal` →
`train-proto` → `train-cascade` → `train-stack` (meta-learner) → `triage`
(K-Means) → `bundle` (packages every trained artifact into one file,
`models/ensemble.pkl`).

On Windows, run with UTF-8 output forced to avoid a console encoding crash on
the pipeline's progress banners:

```bash
set PYTHONIOENCODING=utf-8   # PowerShell: $env:PYTHONIOENCODING="utf-8"
python src/satark.py --mode full-ensemble
```

Intermediate per-model artifacts (`lgb_v2.pkl`, `nnpu_v2.pkl`, etc.) are
written during training and consumed by the final `bundle` step. Once
`models/ensemble.pkl` exists, only that one file is needed to serve — the
intermediates can be deleted.

### 4. Generate predictions and alerts

```bash
python src/satark.py --mode predict     # outputs/predictions.csv
python src/satark.py --mode alerts      # outputs/alerts.json (severity-tiered)
```

### 5. Start the API server (optional, single-model prototype path)

```bash
python src/satark.py --mode serve       # Flask API on :5000
```

### 6. Explain a specific account (SHAP)

```bash
python src/satark.py --mode explain --account_id 42
```

## All CLI Modes

`python src/satark.py --mode <mode>`

| Mode | What it does |
|---|---|
| `extract` | Clean raw dataset (drop constant/correlated features, exclude leakage columns) |
| `train` | Train Model B — cost-sensitive LightGBM, leak-free nested CV |
| `train-nnpu` | Train Model A — NNPU (Positive-Unlabeled) learner |
| `anomaly` | Train Model C — ECOD + Deep Isolation Forest (unsupervised) |
| `train-catboost` | Train Model E — CatBoost |
| `train-focal` | Train Model F — focal-loss LightGBM (3-seed averaged) |
| `train-proto` | Train Model G — prototype/metric-learning classifier |
| `train-tabpfn` | Train Model D — TabPFN (optional, requires hosted API or local license) |
| `train-cascade` | Stage-2 precision-refiner — re-ranks the main ensemble's flagged pool |
| `train-stack` | Fit the meta-learner over all trained models' OOF predictions |
| `ablate` | Ablation runs — which base models actually help the stack |
| `triage` | Whole-population K-Means clustering for investigator prioritization |
| `bundle` | Pack every trained artifact into `models/ensemble.pkl` |
| `predict` | Batch-score all accounts |
| `alerts` | Generate severity-tiered alerts from predictions |
| `monitor` | Model performance / drift check |
| `explain` / `explain-export` | SHAP explanation for one account, or batch-export for the dashboard |
| `serve` | Start the Flask REST API (single-model prototype path — see `production/` for the real serving stack) |
| `retrain` | Continuous retraining with investigator feedback |
| `behavior` | Mule-only behavioral sub-clustering |
| `temporal` | Time-based pattern analysis |
| `stream` | Real-time streaming inference pipeline |
| `active` | Uncertainty sampling for active learning |
| `str` | Auto-generate Suspicious Transaction Reports (STR) |
| `validate-external` | Runs the same methodology against an independent public fraud benchmark to check the method generalizes beyond this dataset's 81 mules |
| `full` | Phase 1 pipeline: extract → train → predict → alerts → monitor |
| `full-ensemble` | Everything above (all models, cascade, meta-learner, triage, bundle) |

## File Structure

```
BOI HACKATHON/
├── src/
│   └── satark.py              # Prototype pipeline: feature engineering, all base
│                               #   models, cascade, meta-learner, clustering,
│                               #   single-model API, SHAP, streaming, STR generation
├── data/
│   ├── DataSet (1).csv        # Official competition dataset (9,082 accounts)
│   └── Description.xlsx       # Official feature dictionary (3,924 features)
├── models/
│   └── ensemble.pkl           # All trained models + scaler + feature list +
│                               #   cascade + clustering model, bundled into one file
├── outputs/
│   ├── features.csv           # Cleaned, label-independent feature matrix (raw
│   │                           #   input every model selects/engineers from)
│   ├── features_lgb_deploy.csv # LightGBM's own final engineered+scaled matrix —
│   │                           #   kept separate so it never overwrites features.csv
│   ├── oof.npz                # Out-of-fold predictions per base model
│   ├── predictions.csv        # Per-account mule scores
│   ├── clusters.csv           # Cluster-level mule density summary
│   └── assignments.csv        # Per-account cluster assignment
├── dashboard/
│   ├── landing.html           # Public-facing landing/pitch page
│   ├── investigator.html      # Investigator-facing console (calls production/serving/api.py)
│   └── assets/                # Images used by the dashboard pages
├── production/                # The real deployment architecture — see below
├── docs/
│   ├── OLD_UNWANTED_final_submission_round2.html  # Archived Round-2 IDE submission
│   │                           #   (2 months old, numbers superseded — do not update)
│   ├── PITCH_VALIDATION_CHECKLIST.md  # Pitch-deck checklist built from analyzing
│   │                           #   official PSB Hackathon Series 2025 winners
│   └── Prototype_Submission/  # IEEE-format prototype paper (LaTeX)
├── requirements.txt
└── README.md                  # This file
```

### `production/` — the deployment architecture

Every module here follows the same pattern: an abstract interface with a real,
ready-to-deploy implementation (needs live infrastructure this environment
doesn't have) plus a fully-tested in-memory/mock implementation that proves the
wiring works today.

| Path | Purpose |
|---|---|
| `feature_pipeline/finacle_adapter.py` | Core-banking ingestion — real Finacle batch-file adapter + mock |
| `feature_pipeline/feature_store.py` | Train/serve-consistent rolling features — Redis (production) + in-memory (dev) |
| `feature_pipeline/schema_mapper.py` | Translates live rolling features into the trained ensemble's exact ~3,924-column schema |
| `feature_pipeline/replay_adapter.py` | Streams real historical accounts (not random mock data) through the live pipeline for demos |
| `feature_pipeline/cross_bank_adapter.py` | DPIP / I4C Suspect Registry signal — the cross-bank blind-spot fix, mocked pending real access |
| `serving/event_bus.py` | Real Kafka event bus + in-memory equivalent |
| `serving/transaction_consumer.py` | Wires ingestion → feature update → scoring → governance action together |
| `serving/api.py`, `serving/auth.py` | Authenticated, role-gated REST API |
| `governance/kill_switch.py`, `human_override.py`, `customer_disclosure.py` | Reversible-by-design automated actions, full audit trail |
| `training/model_registry.py` | Versioned model artifacts, blocks silent regressions |
| `monitoring/drift_monitor.py` | Population Stability Index drift checks |
| `tests/` | 73 tests covering every module above, plus a leak-free-methodology regression guard |

## Reproducibility Notes

- **Leakage-free by construction**: mutual-information feature selection, log/
  polynomial/interaction engineering, and scaling are fit only on each fold's
  training partition and applied — without refitting — to that fold's validation
  rows. The reported metrics are out-of-fold, never in-sample.
- **Excluded features**: `F2230` (batch-collection month), `F3912`
  (`FRAUD_SUSPECTED`), and `F3913`/`F3914`/`F3915` (other resolution-status
  flags — all post-investigation labels, not behavioral signals) are dropped
  before training, along with the raw row index.
- **Primary metric**: AUC-PR, not AUC-ROC, since ROC-AUC is known to inflate
  toward ~0.99 for a trivial classifier under 99:1 class imbalance.
- **A real data-corruption bug was found and fixed (2026-08-20)**: LightGBM's
  training step was saving its own final engineered+scaled matrix back into
  the shared `outputs/features.csv`, causing every model trained after it to
  silently select features from LightGBM's already-reduced space instead of
  the intended raw ~1,841-column matrix. Fixed by giving LightGBM's deployment
  matrix its own path (`outputs/features_lgb_deploy.csv`). Verified via: full
  retrain, 73/73 tests passing, and an independent clean-room re-derivation
  (separate script, fresh 70/30 split, zero shared code) landing in the same
  performance range as the pipeline's own reported numbers.
- **Two other leaks were found and fixed earlier in development** (in-sample
  meta-learner scoring; NNPU reusing a globally-selected feature list instead
  of per-fold selection) — both inflated reported numbers rather than
  breaking outright. `production/tests/test_leak_free_methodology.py` pins a
  tolerance band around the current honest value specifically to catch either
  pattern reappearing silently in the future.
