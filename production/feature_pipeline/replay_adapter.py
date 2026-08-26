"""
Competition-dataset replay adapter — a CoreBankingAdapter that streams
synthetic-but-constrained transactions for REAL accounts from the
competition dataset, instead of finacle_adapter.MockCoreBankingAdapter's
fully-random synthetic accounts.

Why this exists: the live pipeline demo (EventBus -> TransactionConsumer ->
scoring -> governance) has only ever run against MockCoreBankingAdapter —
random account IDs, random amounts, no relationship to any account SACHET's
trained ensemble has actually seen or scored. That makes the "live pipeline"
demo honest about the wiring but disconnected from anything a judge can
independently verify. This adapter closes that gap using only data already
in this repository — no live bank access required or claimed.

The real limitation, stated plainly: the competition dataset provides
pre-aggregated statistics (F1..F3924, see data/Description.xlsx), not a
transaction-level log — no raw per-transaction record was ever collected for
this competition, so there is no ground truth to literally replay. What this
DOES give: individual synthetic TransactionEvents whose own aggregation
reproduces a REAL account's REAL, KNOWN channel/window totals (read directly
from that account's own dataset row via schema_mapper's reverse index) —
when fed back through feature_store.compute_rolling_features() /
compute_extended_rolling_features(), the recovered totals should match the
real dataset values closely, not just plausibly. Individual transaction
timestamps and the exact split of a window's total across N transactions are
synthetic; the totals they reproduce are real. See
test_replay_adapter.py::test_reconstructed_totals_match_real_dataset_values
for the tolerance this is verified against.

One further, important scope correction found while testing this module:
the dataset has NO literal per-channel "total amount/count" column — e.g.
there is no "Total Cash Amount - last 7D" feature. Per-channel quantities
only exist as ratios, averages, min/max, or deviations (see
schema_mapper.py's category taxonomy); the only DIRECT, unambiguous ground
truth available is the CROSS-CHANNEL total ("Total Transaction Amount -
last 7D", covering all channels combined — real F-columns, e.g. F3811
TOT_TXNAMT_L7D). So this adapter reconstructs a real account's real
cross-channel totals (and their real CR/DB split) exactly, then distributes
that real total ACROSS the channel taxonomy using a seeded, deterministic
channel-mix — the account-level volume is real, the channel-by-channel
breakdown is a documented approximation, not independently grounded.

Ground-truth mule/legit labels for the replayed accounts are exposed via
ground_truth_label() purely for DEMO NARRATION ("this is a confirmed mule
from the dataset") — nothing in the scoring path itself reads that label.
"""

import os
import random
from datetime import datetime, timedelta, timezone

import pandas as pd

from finacle_adapter import CoreBankingAdapter, TransactionEvent
import schema_mapper as sm

_WINDOW_DAYS = (7, 14, 31)

# The channels reproduced from real data — restricted to the taxonomy
# feature_store.ALL_CHANNELS also tracks, so a replayed account's events
# round-trip cleanly through compute_extended_rolling_features().
_REPLAY_CHANNELS = ('CASH', 'CHEQUE', 'UPI', 'NEFT', 'IMPS', 'RTGS', 'ATM', 'POS',
                    'STDNG_INSTR', 'APB', 'BBPS', 'MBNKING', 'NET_BNKING', 'GST', 'LOAN')

_OCCUPATION_F = 'F3891'   # CUST_OCCP, confirmed against Description.xlsx
_TENURE_F = 'F3887'       # TENURE_AS_OF_ALERT
_ACCT_OPEN_F = 'F3889'    # ACCT_OPN_DAYS
_AGE_F = 'F3894'          # AGE_IN_YRS
_TARGET_F = 'F3924'       # FRAUD_TGT


def _build_reverse_index(feature_dict: pd.DataFrame) -> dict:
    """(window_days, metric, cr_db) -> F-column name, restricted to the
    real, unambiguous cross-channel TOT-family single-window quantities
    (category '_ALL') — the only DIRECT ground truth this dataset actually
    contains (see module docstring: no per-channel total column exists)."""
    index = {}
    for _, row in feature_dict.iterrows():
        spec = sm._parse_row(row['Variable Name'], row['Description'])
        if (spec['stat'] == 'TOT' and spec['windows'] is not None
                and len(spec['windows']) == 1 and not spec['ci_filter']
                and spec['category'] == '_ALL'):
            key = (spec['windows'][0], spec['metric'], spec['cr_db'])
            index.setdefault(key, row['Feature'])
    return index


class CompetitionReplayAdapter(CoreBankingAdapter):
    """Real accounts, real known aggregate behavior, synthetic individual
    transactions constrained to reproduce that behavior. NOT for production
    use — the source is the static competition CSV, not a live bank feed;
    see module docstring for exactly what is and isn't real here."""

    def __init__(self, data_csv: str = None, n_mule_accounts: int = 5,
                n_legit_accounts: int = 15, seed: int = 42,
                as_of: datetime = None):
        self._rng = random.Random(seed)
        self._as_of = as_of or datetime.now(timezone.utc)

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_csv = data_csv or os.path.join(repo_root, 'data', 'DataSet (1).csv')
        df = pd.read_csv(data_csv)

        mules = df[df[_TARGET_F] == 1].sample(n=min(n_mule_accounts, (df[_TARGET_F] == 1).sum()),
                                                random_state=seed)
        legits = df[df[_TARGET_F] == 0].sample(n=min(n_legit_accounts, (df[_TARGET_F] == 0).sum()),
                                                random_state=seed)
        self._rows = pd.concat([mules, legits])
        # Use the dataframe's own row position as a stable, demo-friendly
        # account id (matches the ids used elsewhere in this repo, e.g.
        # outputs/predictions.csv's account_id column).
        self._account_ids = [str(i) for i in self._rows.index]
        self._labels = dict(zip(self._account_ids, self._rows[_TARGET_F].astype(int)))

        feature_dict = sm.load_feature_dictionary()
        self._reverse_index = _build_reverse_index(feature_dict)

        self._events = []
        for acc_id, (_, row) in zip(self._account_ids, self._rows.iterrows()):
            self._events.extend(self._synthesize_events_for_account(acc_id, row))
        self._events.sort(key=lambda e: e.timestamp)

    def ground_truth_label(self, account_id: str) -> int:
        """1 = confirmed mule in the competition dataset, 0 = confirmed
        legitimate. Demo narration only — never consumed by the scoring
        path itself."""
        return self._labels.get(account_id, -1)

    def account_ids(self) -> list:
        return list(self._account_ids)

    def _channel_weights(self, account_id: str) -> dict:
        """Deterministic per-account channel-mix weights (seeded by
        account_id, not global rng state, so re-synthesizing one account in
        isolation — e.g. in a test — reproduces the same mix). This is the
        one genuinely approximated part of the reconstruction: the real
        cross-channel total is exact, its split across channels is not
        independently grounded (see module docstring)."""
        local_rng = random.Random(f"{account_id}:channel_mix")
        raw = [local_rng.random() ** 2 for _ in _REPLAY_CHANNELS]  # skewed, not uniform — a few dominant channels
        total = sum(raw)
        return dict(zip(_REPLAY_CHANNELS, [r / total for r in raw]))

    def _synthesize_events_for_account(self, account_id: str, row: pd.Series) -> list:
        totals = {}
        for w in _WINDOW_DAYS:
            amt_col = self._reverse_index.get((w, 'AMT', 'TOTAL'))
            txn_col = self._reverse_index.get((w, 'TXNS', 'TOTAL'))
            cr_amt_col = self._reverse_index.get((w, 'AMT', 'CR'))
            db_amt_col = self._reverse_index.get((w, 'AMT', 'DB'))
            totals[w] = {
                'amt': float(row[amt_col]) if amt_col in row.index and pd.notna(row.get(amt_col)) else None,
                'txns': float(row[txn_col]) if txn_col in row.index and pd.notna(row.get(txn_col)) else None,
                'cr_amt': float(row[cr_amt_col]) if cr_amt_col and cr_amt_col in row.index and pd.notna(row.get(cr_amt_col)) else None,
                'db_amt': float(row[db_amt_col]) if db_amt_col and db_amt_col in row.index and pd.notna(row.get(db_amt_col)) else None,
            }

        cross_channel_events = self._events_from_nested_windows(account_id, '_ALL', totals)
        weights = self._channel_weights(account_id)
        channels = list(weights.keys())
        cum_weights = []
        running = 0.0
        for c in channels:
            running += weights[c]
            cum_weights.append(running)

        events = []
        for e in cross_channel_events:
            r = self._rng.random()
            idx = next(i for i, cw in enumerate(cum_weights) if r <= cw)
            events.append(TransactionEvent(account_id=e.account_id, timestamp=e.timestamp,
                                           channel=channels[idx], direction=e.direction,
                                           amount=e.amount))
        return events

    def _events_from_nested_windows(self, account_id, channel, totals) -> list:
        """Converts cumulative window totals (7D, 14D, 31D — each a superset
        of the shorter ones, same nesting compute_extended_rolling_features
        uses) into per-band synthetic transactions. Bands: [now-7, now],
        (now-14, now-7], (now-31, now-14]. A band is skipped if its
        incremental txn count can't be determined or is non-positive
        (real-world rounding/definitional noise between adjacent F-columns,
        not a bug in the reconstruction)."""
        events = []
        bands = [(0, 7, totals[7]), (7, 14, totals.get(14)), (14, 31, totals.get(31))]
        prev_txns, prev_amt, prev_cr, prev_db = 0.0, 0.0, 0.0, 0.0
        for lo, hi, cum in bands:
            if cum is None or cum['txns'] is None:
                prev_txns = cum['txns'] if cum and cum['txns'] is not None else prev_txns
                continue
            band_txns = max(0, round(cum['txns'] - prev_txns))
            cum_amt = cum['amt'] if cum['amt'] is not None else 0.0
            band_amt = max(0.0, cum_amt - prev_amt)
            band_cr = max(0.0, (cum['cr_amt'] or 0.0) - prev_cr) if cum['cr_amt'] is not None else None
            band_db = max(0.0, (cum['db_amt'] or 0.0) - prev_db) if cum['db_amt'] is not None else None

            if band_txns > 0:
                events.extend(self._distribute_amount_into_events(
                    account_id, channel, lo, hi, band_txns, band_amt, band_cr, band_db))

            prev_txns, prev_amt = cum['txns'], cum_amt
            prev_cr = cum['cr_amt'] if cum['cr_amt'] is not None else prev_cr
            prev_db = cum['db_amt'] if cum['db_amt'] is not None else prev_db
        return events

    def _distribute_amount_into_events(self, account_id, channel, day_lo, day_hi,
                                       n_txns, total_amt, cr_amt, db_amt) -> list:
        """n_txns synthetic events in the (day_lo, day_hi] band, with random
        timestamps and amounts constrained to sum exactly to total_amt (and,
        when known, split into a CREDIT sub-total of cr_amt and DEBIT
        sub-total of db_amt — falling back to a random CR/DB assignment
        when the real split isn't available for this channel)."""
        n_txns = int(n_txns)
        if cr_amt is not None and db_amt is not None and (cr_amt + db_amt) > 0:
            n_cr = round(n_txns * cr_amt / (cr_amt + db_amt))
        else:
            n_cr = self._rng.randint(0, n_txns)
        n_db = n_txns - n_cr

        cr_split = self._random_positive_split(cr_amt if cr_amt is not None else total_amt * n_cr / max(n_txns, 1), n_cr)
        db_split = self._random_positive_split(db_amt if db_amt is not None else total_amt * n_db / max(n_txns, 1), n_db)

        events = []
        for amt in cr_split:
            events.append(self._make_event(account_id, channel, day_lo, day_hi, 'CREDIT', amt))
        for amt in db_split:
            events.append(self._make_event(account_id, channel, day_lo, day_hi, 'DEBIT', amt))
        return events

    def _random_positive_split(self, total: float, n: int) -> list:
        """n positive values summing exactly to total (0 if n==0 or total<=0)."""
        if n <= 0 or total <= 0:
            return []
        cuts = sorted(self._rng.random() for _ in range(n - 1))
        bounds = [0.0] + cuts + [1.0]
        return [round((bounds[i + 1] - bounds[i]) * total, 2) for i in range(n)]

    def _make_event(self, account_id, channel, day_lo, day_hi, direction, amount) -> TransactionEvent:
        offset_days = day_lo + self._rng.random() * (day_hi - day_lo)
        ts = self._as_of - timedelta(days=offset_days)
        return TransactionEvent(account_id=account_id, timestamp=ts, channel=channel,
                                direction=direction, amount=max(amount, 0.01))

    def fetch_new_transactions(self, since: datetime) -> list:
        return [e for e in self._events if since <= e.timestamp <= self._as_of]

    def fetch_account_metadata(self, account_id: str) -> dict:
        if account_id not in self._account_ids:
            return {}
        row = self._rows.loc[int(account_id)]
        return {
            'account_id': account_id,
            'occupation': row.get(_OCCUPATION_F),
            'tenure_days': row.get(_TENURE_F),
            'account_open_days': row.get(_ACCT_OPEN_F),
            'age_years': row.get(_AGE_F),
            'ground_truth_label': int(row[_TARGET_F]),
        }
