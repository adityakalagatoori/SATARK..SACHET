"""
Schema mapper — translates live rolling features (feature_store.py) into the
competition dataset's exact ~3,924 F-column schema the trained ensemble
expects.

Why this exists: SACHET's trained models were fit on `data/DataSet (1).csv`,
whose columns are anonymized as F1..F3924. The live pipeline
(finacle_adapter.py -> feature_store.py) computes rolling features under a
readable channel/window naming convention. Nothing previously translated
between the two, which is why transaction_consumer.py's scorer has always
been a pluggable callable rather than the real trained ensemble (see that
module's docstring). This module is that translation layer.

How the mapping was derived: `data/Description.xlsx` gives every F-column a
human `Description` (e.g. "Ratio of Cash Credit Amount - last 14 to 31D").
That text was parsed (not guessed from the abbreviated `Variable Name` alone,
which uses ambiguous tokens like CI/TA/MM/DA that the Description text
resolves unambiguously) into a structured spec per feature: which of 16
channel categories, which of 9 statistic families, credit/debit/total, and
which time window(s). See _parse_row() below.

Honesty about coverage and open conventions — read before trusting a number:
  * Every category, statistic family, and window this module recognizes is
    backed by literal phrases actually observed in Description.xlsx (see the
    _CATEGORY_PATTERNS / _STAT_PATTERNS tables). A row whose description
    doesn't match anything gets a value of 0.0 and is counted as UNRESOLVED
    in the coverage report, never silently guessed.
  * Several statistic families have more than one textually-plausible
    reading and a single convention had to be picked. These are marked
    NEEDS_CONFIRMATION in the coverage report and documented at each
    function below:
      - D_  (window-pair "deviation"): implemented as a literal difference,
        shorter window's value minus the longer window's value.
      - DA_ (deviation of averages): same, but on the per-day rate
        (raw total / window length in days) rather than the raw total.
      - D_TA_ (deviation of total from average): implemented as the stated
        window's total minus the 31-day window's per-day rate. The
        Description text for this family is internally inconsistent about
        which windows are involved (see module tests) — treat this family
        as the least-trustworthy of the set.
      - MIN_/MAX_/AVG_/MM_ against a TXNS (count) metric, rather than an
        AMT (amount) metric: per-transaction min/max/average of a *count*
        isn't a well-defined quantity, so these resolve to the window's raw
        transaction count instead (documented placeholder, flagged
        NEEDS_CONFIRMATION) rather than inventing a number.
      - _OCC / _OC suffix (occupation-peer-conditioned deviation): this
        module resolves the underlying single-window quantity but does NOT
        subtract an occupation-peer benchmark (no verified peer-benchmark
        table exists yet — see build_occupation_benchmarks() below, which is
        a real but partial start on one, not a complete solution). These
        features are returned un-adjusted and flagged OCC_NOT_ADJUSTED.
  * The CI_ ("Customer Induced") filter is applied only to the TOTAL
    (not separately to the CR/DB split) statistic, per
    feature_store.compute_extended_rolling_features()'s ci_total_amt /
    ci_total_txns fields — a CR- or DB-specific Customer-Induced feature
    falls back to the unfiltered CR/DB value (flagged
    CI_SPLIT_NOT_AVAILABLE), since the raw feature computation does not
    currently track customer-induced status split by credit/debit.

None of the above is leakage or overfitting risk — it only concerns
reconstructing PRODUCTION-TIME features from live events, entirely separate
from src/satark.py's training pipeline and its own (separately verified,
see production/tests/test_leak_free_methodology.py) leak-free discipline.
"""

import os
import re

import pandas as pd

_WINDOW_DAYS = (7, 14, 31)

_DESC_XLSX = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'data', 'Description.xlsx')

# Ordered, first match wins — more specific phrases must precede substrings
# they contain (e.g. "Non Cash Non Cheque" before "Cash").
_CATEGORY_PATTERNS = [
    ('non cash non cheque', 'NON_CASH_CHQ'),
    ('cash', 'CASH'),
    ('cheque', 'CHEQUE'),
    ('upi', 'UPI'),
    ('online transfer', 'ELEC_XFER'),          # "Online Transfer (IMPS+NEFT+RTGS)"
    ('atm', 'ATM'),
    ('merchant payment', 'POS'),
    ('standing instruction', 'STDNG_INSTR'),
    ('aadhar payment bridge', 'APB'),
    ('aadhaar payment bridge', 'APB'),
    ('bharat bill payment system', 'BBPS'),
    ('mobile banking', 'MBNKING'),
    ('net banking', 'NET_BNKING'),
    ('gst', 'GST'),
    ('loan', 'LOAN'),
    ('customer induced fees charges', 'FEES_CHRGS'),
    ('account balance', '_BAL'),
]

# Ordered, first match wins — matched against the DESCRIPTION PREFIX.
_STAT_PATTERNS = [
    ('ratio of avgs: of', 'RA'),
    ('ratio of', 'R'),
    ('deviation of avgs: of', 'DA'),
    ('deviation of total from avg:', 'D_TA'),
    ('deviation of', 'D'),
    ('difference of max and min:', 'MM'),
    ('min ', 'MIN'),
    ('max ', 'MAX'),
    ('average ', 'AVG'),
    ('total ', 'TOT'),
]

_WINDOW_RE = re.compile(r'last\s+(\d+)\s*(?:to\s*(\d+))?\s*d(?:ays)?', re.I)
_OCC_RE = re.compile(r'_OCC$|_OC$')


def load_feature_dictionary(path: str = None) -> pd.DataFrame:
    """Loads data/Description.xlsx. Cached at module level via functools
    would risk staleness across a long-running process reloading the file;
    callers that call this in a hot loop should cache the DataFrame
    themselves (map_live_features_to_model_schema does)."""
    return pd.read_excel(path or _DESC_XLSX)


def _parse_row(variable_name: str, description: str) -> dict:
    """Parses one Description.xlsx row into a structured spec. Returns a
    dict with keys: category, stat, cr_db, metric, windows (tuple or None),
    occ (bool), ci_filter (bool). category/stat/windows are None when the
    description text doesn't match anything this module recognizes — the
    caller must treat that as UNRESOLVED, not silently default."""
    desc = str(description).strip()
    dlow = desc.lower()

    category = None
    for phrase, cat in _CATEGORY_PATTERNS:
        if phrase in dlow:
            category = cat
            break

    ci_filter = 'customer induced' in dlow
    if category is None and ci_filter:
        category = '_ALL'   # "Customer Induced Credit Amount" with no other category = all channels, CI-filtered

    stat = None
    for phrase, fam in _STAT_PATTERNS:
        if dlow.startswith(phrase):
            stat = fam
            break

    if category is None and stat == 'TOT':
        # "Total Transaction Amount - last 7D" etc., with no channel phrase
        # at all, means the cross-channel grand total — the dataset has no
        # literal per-channel "total" column (only per-channel ratio/
        # average/min/max/deviation ones — see module docstring), so this
        # is the real, unambiguous meaning of a bare TOT_ feature, not a
        # fallback guess.
        category = '_ALL'

    if 'credit' in dlow:
        cr_db = 'CR'
    elif 'debit' in dlow:
        cr_db = 'DB'
    else:
        cr_db = 'TOTAL'

    metric = 'AMT' if ('amount' in dlow or 'balance' in dlow) else 'TXNS'

    m = _WINDOW_RE.search(dlow)
    windows = None
    if m:
        windows = (int(m.group(1)), int(m.group(2))) if m.group(2) else (int(m.group(1)),)

    occ = bool(_OCC_RE.search(str(variable_name)))

    return {
        'category': category, 'stat': stat, 'cr_db': cr_db, 'metric': metric,
        'windows': windows, 'occ': occ, 'ci_filter': ci_filter,
    }


def _subkey(metric: str, cr_db: str, ci_filter: bool) -> tuple:
    """Returns (subkey, needs_ci_split_fallback) — the key into a raw
    per-window stats dict (from compute_extended_rolling_features /
    _amount_stats), and whether a requested CI+CR/DB combination had to fall
    back to the unfiltered value because that split isn't tracked."""
    if metric == 'AMT':
        base = {'TOTAL': 'total_amt', 'CR': 'cr_amt', 'DB': 'db_amt'}[cr_db]
    else:
        base = {'TOTAL': 'total_txns', 'CR': 'cr_txns', 'DB': 'db_txns'}[cr_db]
    if ci_filter:
        if cr_db == 'TOTAL':
            return ('ci_total_amt' if metric == 'AMT' else 'ci_total_txns'), False
        return base, True  # CI+CR/DB split not tracked — fall back, flag it
    return base, False


def _bal_subkey(stat: str, cr_db: str) -> str:
    return {'MIN': 'min_bal', 'MAX': 'max_bal', 'AVG': 'avg_bal', 'TOT': 'total_bal'}.get(stat, 'total_bal')


def compute_feature_value(raw: dict, spec: dict) -> tuple:
    """Computes one F-column's value from the raw per-channel-per-window
    dict (feature_store.compute_extended_rolling_features output) and a
    parsed spec (from _parse_row). Returns (value, status) where status is
    one of: 'OK', 'UNRESOLVED', 'NEEDS_CONFIRMATION', 'OCC_NOT_ADJUSTED',
    'CI_SPLIT_NOT_AVAILABLE'."""
    category, stat, cr_db, metric = spec['category'], spec['stat'], spec['cr_db'], spec['metric']
    windows = spec['windows']

    if category is None or stat is None or windows is None:
        return 0.0, 'UNRESOLVED'
    if category not in raw:
        return 0.0, 'UNRESOLVED'

    status = 'OK'

    if category == '_BAL':
        subkey = _bal_subkey(stat, cr_db)
        if stat in ('MIN', 'MAX', 'AVG', 'TOT') and len(windows) == 1:
            w = windows[0]
            return raw['_BAL'].get(w, {}).get(subkey, 0.0), status
        if len(windows) == 2:
            a, b = windows
            va = raw['_BAL'].get(a, {}).get(subkey, 0.0)
            vb = raw['_BAL'].get(b, {}).get(subkey, 0.0)
            return va + vb if stat == 'TOT' else va - vb, 'NEEDS_CONFIRMATION'
        return 0.0, 'UNRESOLVED'

    subkey, ci_fallback = _subkey(metric, cr_db, spec['ci_filter'])
    if ci_fallback:
        status = 'CI_SPLIT_NOT_AVAILABLE'

    def get(w):
        return raw.get(category, {}).get(w, {}).get(subkey, 0.0)

    def days(w):
        return float(w)

    if len(windows) == 1:
        w = windows[0]
        if stat == 'TOT':
            return get(w), status
        if stat == 'MIN':
            if metric == 'AMT':
                return raw.get(category, {}).get(w, {}).get('min_amt', 0.0), status
            return get(w), 'NEEDS_CONFIRMATION'  # min-of-count placeholder
        if stat == 'MAX':
            if metric == 'AMT':
                return raw.get(category, {}).get(w, {}).get('max_amt', 0.0), status
            return get(w), 'NEEDS_CONFIRMATION'
        if stat == 'AVG':
            if metric == 'AMT':
                return raw.get(category, {}).get(w, {}).get('avg_amt', 0.0), status
            return get(w) / days(w), status  # avg per day for a count metric
        if stat == 'MM':
            if metric == 'AMT':
                d = raw.get(category, {}).get(w, {})
                return d.get('max_amt', 0.0) - d.get('min_amt', 0.0), status
            return 0.0, 'NEEDS_CONFIRMATION'
        if spec['occ'] and stat in ('D', 'DA'):
            # Single-window "deviation" + _OCC/_OC suffix = deviation from an
            # occupation-peer benchmark, not a window-pair difference. No
            # verified benchmark table is wired in yet (see
            # build_occupation_benchmarks below) so the raw value is
            # returned un-adjusted.
            val = get(w) if stat == 'D' else get(w) / days(w)
            return val, 'OCC_NOT_ADJUSTED'
        return 0.0, 'UNRESOLVED'

    if len(windows) == 2:
        a, b = windows  # a is the shorter window, b the longer, per Description.xlsx's own ordering
        if stat == 'R':
            denom = get(b)
            return (get(a) / denom) if denom else 0.0, status
        if stat == 'RA':
            ra, rb = get(a) / days(a), get(b) / days(b)
            return (ra / rb) if rb else 0.0, status
        if stat == 'D':
            return get(a) - get(b), 'NEEDS_CONFIRMATION'
        if stat == 'DA':
            return (get(a) / days(a)) - (get(b) / days(b)), 'NEEDS_CONFIRMATION'
        if stat == 'D_TA':
            # "Deviation of total from avg" — Description.xlsx's own text is
            # internally inconsistent about which windows are involved (see
            # module docstring). Best-effort reading: the shorter window's
            # total minus the longest (31D) window's per-day rate.
            longest = max(_WINDOW_DAYS)
            return get(a) - (raw.get(category, {}).get(longest, {}).get(subkey, 0.0) / longest), 'NEEDS_CONFIRMATION'
        if stat == 'TOT':
            return get(a) + get(b), 'NEEDS_CONFIRMATION'
        return 0.0, 'UNRESOLVED'

    return 0.0, 'UNRESOLVED'


def map_live_features_to_model_schema(events: list, account_id: str, as_of=None,
                                      feature_dict: pd.DataFrame = None) -> tuple:
    """End-to-end: live TransactionEvents for one account -> a full,
    model-schema-shaped feature dict {F-number: value}, plus a coverage
    report. Does not touch training data or model weights — purely a
    production-time feature-reconstruction step (see module docstring for
    the leak-free/train-serve-consistency boundary that keeps separate)."""
    from feature_store import compute_extended_rolling_features

    raw = compute_extended_rolling_features(account_id, events, as_of=as_of)
    fd = feature_dict if feature_dict is not None else load_feature_dictionary()

    values = {}
    status_counts = {}
    unresolved_examples = []
    for _, row in fd.iterrows():
        fcol = row['Feature']
        if fcol in ('F3924',):  # never reconstruct the target itself
            continue
        spec = _parse_row(row['Variable Name'], row['Description'])
        val, status = compute_feature_value(raw, spec)
        values[fcol] = val
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == 'UNRESOLVED' and len(unresolved_examples) < 10:
            unresolved_examples.append((fcol, row['Variable Name']))

    total = sum(status_counts.values())
    coverage = {
        'total_features': total,
        'status_counts': status_counts,
        'resolved_fraction': (total - status_counts.get('UNRESOLVED', 0)) / total if total else 0.0,
        'exact_confidence_fraction': status_counts.get('OK', 0) / total if total else 0.0,
        'unresolved_examples': unresolved_examples,
    }
    return values, coverage


def build_occupation_benchmarks(features_csv: str, data_csv: str) -> dict:
    """Partial, working start on an occupation-peer benchmark table for the
    _OCC/_OC-suffixed deviation features (see module docstring — those
    currently return un-adjusted values). Precomputes the historical
    per-occupation MEAN of every DIRECT (non-ratio, single-window) F-column
    this module can already parse, from the static competition dataset —
    the only quantities we have verified ground truth for. Ratio/deviation
    F-columns are NOT benchmarked here (their historical per-occupation mean
    isn't a meaningful "peer norm" the same way a raw total is) — this is
    explicitly a partial mechanism, not full OCC-adjustment coverage.

    Requires a 'CUST_OCCP' (or similarly named occupation) column in the
    historical dataset; returns an empty dict with a clear reason if that
    column isn't present rather than guessing at a substitute.
    """
    df = pd.read_csv(data_csv)
    occ_col = next((c for c in df.columns if c.upper() in ('CUST_OCCP', 'OCCUPATION')), None)
    if occ_col is None:
        return {'_error': 'no occupation column found in data_csv — benchmarks not built'}

    fd = load_feature_dictionary()
    benchmarks = {}
    for _, row in fd.iterrows():
        fcol = row['Feature']
        if fcol not in df.columns:
            continue
        spec = _parse_row(row['Variable Name'], row['Description'])
        if spec['stat'] not in ('TOT', 'MIN', 'MAX', 'AVG') or spec['windows'] is None or len(spec['windows']) != 1:
            continue  # only direct, single-window quantities are meaningful peer benchmarks
        benchmarks[fcol] = df.groupby(occ_col)[fcol].mean().to_dict()

    return benchmarks
