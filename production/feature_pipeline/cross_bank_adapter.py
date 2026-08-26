"""
Cross-bank signal adapter — DPIP / I4C Suspect Registry.

Why this exists: the earlier deep-dive into SACHET's 10 undetectable mules
found that 2 of them are a genuine blind spot — high-value, non-Savings
accounts that look statistically identical to legitimate large accounts
using only single-account transaction data. External research at the time
concluded this typology structurally requires counterparty/network data,
which this dataset does not contain.

This module closes that gap architecturally. Two pieces of real Indian
banking infrastructure now exist for exactly this signal:

1. RBI's Digital Payments Intelligence Platform (DPIP) — launched 2025 with
   NPCI, a nationwide real-time data-sharing network letting banks and
   fintechs share verified signals about suspicious accounts/transactions
   ACROSS institutions — the cross-bank validation this dataset structurally
   lacks.
2. I4C's Suspect Registry, integrated with RBI Innovation Hub's MuleHunter.ai
   since a May 2026 MoU — cross-agency intelligence on suspicious accounts
   and cyber-fraud networks, shared with banks' AI detection systems.

Neither is accessible from this hackathon environment (both require formal
integration agreements with RBI/NPCI/I4C that only a live bank deployment
could obtain) — this module defines the adapter CONTRACT so a production
deployment can plug into either the moment BOI's integration is live,
matching the same pattern as finacle_adapter.py. The mock implementation
exists purely to demonstrate and test how this signal would be consumed by
the rest of the pipeline, not to claim real cross-bank data.

RBI context: MuleHunter.ai (RBI Innovation Hub's own tool) is already
deployed across 26+ banks, reporting ~95% accuracy at Canara Bank with ~90%
positive-alert precision, built on 19 documented mule-behavior patterns.
SACHET is not positioned as a replacement for MuleHunter.ai — it is a
complementary, tiered, explainable, bank-specific layer; where MuleHunter.ai
already runs, this cross-bank signal (once real access exists) would be one
of its natural inputs alongside SACHET's own account-level scoring.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
import random


@dataclass
class CrossBankSignal:
    account_id: str
    flagged_in_suspect_registry: bool
    counterparty_flagged_count: int   # how many of this account's recent counterparties are themselves flagged elsewhere
    cross_bank_velocity_score: float  # 0-1, how unusually fast funds move OUT to other institutions
    source: str                       # 'DPIP' or 'I4C_SUSPECT_REGISTRY' or 'NONE'
    as_of: datetime


class CrossBankAdapter(ABC):
    @abstractmethod
    def fetch_signal(self, account_id: str, counterparty_account_ids: list[str] = None) -> CrossBankSignal:
        raise NotImplementedError


class DPIPAdapter(CrossBankAdapter):
    """Real integration point — requires BOI to be an onboarded DPIP
    participant with NPCI/RBI. Not implementable or testable without that
    formal access; this class defines the exact contract to implement once
    it exists, so the rest of the pipeline never needs to change."""

    def __init__(self, api_endpoint: str, api_credentials: dict):
        self.api_endpoint = api_endpoint
        self.api_credentials = api_credentials
        raise NotImplementedError(
            "DPIP API access requires a formal NPCI/RBI integration agreement "
            "not available in this environment — this constructor intentionally "
            "raises so a stub connection is never silently used as if it were real"
        )

    def fetch_signal(self, account_id: str, counterparty_account_ids: list[str] = None) -> CrossBankSignal:
        raise NotImplementedError


class MockCrossBankAdapter(CrossBankAdapter):
    """Deterministic mock so the rest of the pipeline (feature computation,
    scoring, governance) can be built and tested against this signal shape
    before real DPIP/I4C access exists. NOT for production use — every value
    returned is synthetic."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def fetch_signal(self, account_id: str, counterparty_account_ids: list[str] = None) -> CrossBankSignal:
        counterparty_account_ids = counterparty_account_ids or []
        flagged = self._rng.random() < 0.02  # mule accounts are rare — mock at a realistic base rate
        return CrossBankSignal(
            account_id=account_id,
            flagged_in_suspect_registry=flagged,
            counterparty_flagged_count=self._rng.randint(0, 2) if not flagged else self._rng.randint(1, 5),
            cross_bank_velocity_score=self._rng.random() if not flagged else round(self._rng.uniform(0.6, 1.0), 3),
            source='MOCK',
            as_of=datetime.now(timezone.utc),
        )


def signal_to_features(signal: CrossBankSignal) -> dict:
    """Convert a CrossBankSignal into the flat feature dict shape the rest
    of the feature pipeline (feature_store.py) expects, so it can be merged
    directly into an account's feature row alongside the rolling
    transaction features."""
    return {
        'XBANK_flagged_in_registry': int(signal.flagged_in_suspect_registry),
        'XBANK_counterparty_flagged_count': signal.counterparty_flagged_count,
        'XBANK_velocity_score': signal.cross_bank_velocity_score,
    }
