"""Regression tests for schema_mapper.py — the live-feature-to-model-schema
translation layer (see that module's docstring for the honesty/coverage
caveats these tests exist to pin down, not paper over)."""

import os
import sys
from datetime import datetime, timezone, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'feature_pipeline'))

from finacle_adapter import TransactionEvent
from feature_store import compute_extended_rolling_features
import schema_mapper as sm

_DATA_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         'data', 'Description.xlsx')

pytestmark = pytest.mark.skipif(not os.path.exists(_DATA_CSV), reason="data/Description.xlsx not present")


# ---------------------------------------------------------------------------
# Parser correctness — pinned against actual rows read from Description.xlsx
# during development, so a change to the parsing rules can't silently
# misclassify a real feature without a test failing.
# ---------------------------------------------------------------------------

def test_parse_simple_ratio():
    spec = sm._parse_row('R_CASH_TXN_L7_14D', 'Ratio of Cash Total txns - last 7 to 14D')
    assert spec == {'category': 'CASH', 'stat': 'R', 'cr_db': 'TOTAL', 'metric': 'TXNS',
                    'windows': (7, 14), 'occ': False, 'ci_filter': False}


def test_parse_credit_amount_ratio():
    spec = sm._parse_row('R_CASH_AMT_CR_L14_31D', 'Ratio of Cash Credit Amount - last 14 to 31D')
    assert spec['category'] == 'CASH'
    assert spec['stat'] == 'R'
    assert spec['cr_db'] == 'CR'
    assert spec['metric'] == 'AMT'
    assert spec['windows'] == (14, 31)


def test_parse_ratio_of_averages():
    spec = sm._parse_row('RA_STDNG_INSTR_AMT_L7_14D',
                         'Ratio of avgs: of Standing Instruction Total Amount - last 7 to 14D')
    assert spec['category'] == 'STDNG_INSTR'
    assert spec['stat'] == 'RA'


def test_parse_min_single_window():
    spec = sm._parse_row('MIN_APB_AMT_DB_L7D', 'Min Aadhar Payment Bridge Debit Amount - last 7D')
    assert spec['category'] == 'APB'
    assert spec['stat'] == 'MIN'
    assert spec['cr_db'] == 'DB'
    assert spec['windows'] == (7,)


def test_parse_elec_xfer_combined_category():
    spec = sm._parse_row('MAX_ELEC_XFER_AMT_CR_L31D',
                         'Max Online Transfer (IMPS+NEFT+RTGS) Credit Amount - last 31D')
    assert spec['category'] == 'ELEC_XFER'


def test_parse_max_minus_min():
    spec = sm._parse_row('MM_GST_TXNS_DB_L7D', 'Difference of Max and Min: GST Debit Txns - last 7D')
    assert spec['category'] == 'GST'
    assert spec['stat'] == 'MM'


def test_parse_customer_induced_no_other_category():
    spec = sm._parse_row('CI_AMT_CR_L7D', 'Customer Induced Credit Amount - last 7D')
    assert spec['category'] == '_ALL'
    assert spec['ci_filter'] is True


def test_parse_customer_induced_with_category():
    spec = sm._parse_row('R_CI_NON_CASH_CHQ_TXN_L7_14D',
                         'Ratio of Customer Induced Non Cash Non Cheque Total txns - last 7 to 14D')
    assert spec['category'] == 'NON_CASH_CHQ'
    assert spec['ci_filter'] is True


def test_parse_deviation_window_pair():
    spec = sm._parse_row('D_LOAN_AMT_L14_31D', 'Deviation of Loan Total Amount - last 14 to 31D')
    assert spec['category'] == 'LOAN'
    assert spec['stat'] == 'D'
    assert spec['windows'] == (14, 31)
    assert spec['occ'] is False


def test_parse_occ_suffixed_single_window():
    spec = sm._parse_row('D_CASH_TXN_7D_OCC', 'Deviation of Cash Total txns - last 7D')
    assert spec['category'] == 'CASH'
    assert spec['stat'] == 'D'
    assert spec['windows'] == (7,)
    assert spec['occ'] is True


def test_parse_balance_category():
    spec = sm._parse_row('TOT_BAL_7DAYS', 'Total  account balance in last 7 days')
    assert spec['category'] == '_BAL'
    assert spec['windows'] == (7,)


def test_parse_unresolved_returns_none_fields_not_guesses():
    spec = sm._parse_row('SOME_UNKNOWN_XYZ', 'Something entirely unrecognized happened here')
    assert spec['category'] is None
    assert spec['stat'] is None


# ---------------------------------------------------------------------------
# Value computation — hand-computable small examples
# ---------------------------------------------------------------------------

def _events():
    now = datetime.now(timezone.utc)
    return [
        TransactionEvent('A1', now - timedelta(days=2), 'CASH', 'CREDIT', 1000),
        TransactionEvent('A1', now - timedelta(days=3), 'CASH', 'DEBIT', 400),
        TransactionEvent('A1', now - timedelta(days=10), 'CASH', 'CREDIT', 5000),
        TransactionEvent('A1', now - timedelta(days=1), 'UPI', 'CREDIT', 200, is_customer_induced=True),
        TransactionEvent('A1', now - timedelta(days=1), 'NEFT', 'CREDIT', 300),
        TransactionEvent('A1', now - timedelta(days=1), 'IMPS', 'CREDIT', 100),
    ], now


def test_extended_raw_features_basic_totals():
    events, now = _events()
    raw = compute_extended_rolling_features('A1', events, as_of=now)
    assert raw['CASH'][7]['total_amt'] == 1400          # the two recent CASH txns, not the 10-day-old one
    assert raw['CASH'][31]['total_amt'] == 6400          # all three
    assert raw['CASH'][7]['cr_amt'] == 1000
    assert raw['CASH'][7]['db_amt'] == 400


def test_extended_raw_features_elec_xfer_combines_neft_imps_rtgs():
    events, now = _events()
    raw = compute_extended_rolling_features('A1', events, as_of=now)
    assert raw['ELEC_XFER'][7]['total_amt'] == 400   # NEFT 300 + IMPS 100


def test_extended_raw_features_non_cash_chq_excludes_cash_and_cheque():
    events, now = _events()
    raw = compute_extended_rolling_features('A1', events, as_of=now)
    # UPI(200) + NEFT(300) + IMPS(100) = 600, CASH/CHEQUE excluded
    assert raw['NON_CASH_CHQ'][7]['total_amt'] == 600


def test_extended_raw_features_min_max_avg():
    events, now = _events()
    raw = compute_extended_rolling_features('A1', events, as_of=now)
    cash7 = raw['CASH'][7]
    assert cash7['min_amt'] == 400
    assert cash7['max_amt'] == 1000
    assert cash7['avg_amt'] == 700


def test_compute_ratio_value():
    events, now = _events()
    raw = compute_extended_rolling_features('A1', events, as_of=now)
    spec = {'category': 'CASH', 'stat': 'R', 'cr_db': 'TOTAL', 'metric': 'AMT',
            'windows': (7, 31), 'occ': False, 'ci_filter': False}
    val, status = sm.compute_feature_value(raw, spec)
    assert val == pytest.approx(1400 / 6400)
    assert status == 'OK'


def test_compute_ratio_zero_denominator_does_not_crash():
    events, now = _events()
    raw = compute_extended_rolling_features('A1', events, as_of=now)
    spec = {'category': 'GST', 'stat': 'R', 'cr_db': 'TOTAL', 'metric': 'AMT',
            'windows': (7, 31), 'occ': False, 'ci_filter': False}
    val, status = sm.compute_feature_value(raw, spec)
    assert val == 0.0
    assert status == 'OK'


def test_compute_unresolved_spec_returns_default_zero():
    val, status = sm.compute_feature_value({}, {'category': None, 'stat': None, 'cr_db': 'TOTAL',
                                                  'metric': 'AMT', 'windows': None, 'occ': False, 'ci_filter': False})
    assert val == 0.0
    assert status == 'UNRESOLVED'


def test_compute_occ_flag_is_surfaced_not_hidden():
    events, now = _events()
    raw = compute_extended_rolling_features('A1', events, as_of=now)
    spec = {'category': 'CASH', 'stat': 'D', 'cr_db': 'TOTAL', 'metric': 'AMT',
            'windows': (7,), 'occ': True, 'ci_filter': False}
    val, status = sm.compute_feature_value(raw, spec)
    assert status == 'OCC_NOT_ADJUSTED'


def test_compute_mm_max_minus_min():
    events, now = _events()
    raw = compute_extended_rolling_features('A1', events, as_of=now)
    spec = {'category': 'CASH', 'stat': 'MM', 'cr_db': 'TOTAL', 'metric': 'AMT',
            'windows': (7,), 'occ': False, 'ci_filter': False}
    val, status = sm.compute_feature_value(raw, spec)
    assert val == 1000 - 400


def test_compute_balance_window():
    now = datetime.now(timezone.utc)
    events = [
        TransactionEvent('A1', now - timedelta(days=1), 'CASH', 'CREDIT', 100, balance_after=5000),
        TransactionEvent('A1', now - timedelta(days=2), 'CASH', 'CREDIT', 100, balance_after=4900),
    ]
    raw = compute_extended_rolling_features('A1', events, as_of=now)
    spec = {'category': '_BAL', 'stat': 'MAX', 'cr_db': 'TOTAL', 'metric': 'AMT',
            'windows': (7,), 'occ': False, 'ci_filter': False}
    val, status = sm.compute_feature_value(raw, spec)
    assert val == 5000


# ---------------------------------------------------------------------------
# End-to-end against the REAL Description.xlsx — measures actual coverage,
# doesn't assert a hoped-for number.
# ---------------------------------------------------------------------------

def test_end_to_end_mapping_and_coverage_report():
    events, now = _events()
    values, coverage = sm.map_live_features_to_model_schema(events, 'A1', as_of=now)
    assert coverage['total_features'] > 0
    assert 0.0 <= coverage['resolved_fraction'] <= 1.0
    assert 'F1' in values           # first real feature column reconstructed
    assert 'F3924' not in values    # target column never reconstructed
    print(f"\n[schema_mapper coverage] resolved={coverage['resolved_fraction']:.1%} "
          f"exact_confidence={coverage['exact_confidence_fraction']:.1%} "
          f"breakdown={coverage['status_counts']}")


def test_end_to_end_no_crash_on_account_with_zero_events():
    values, coverage = sm.map_live_features_to_model_schema([], 'GHOST', as_of=datetime.now(timezone.utc))
    assert coverage['total_features'] > 0
    assert all(v == 0.0 or isinstance(v, float) for v in list(values.values())[:50])
