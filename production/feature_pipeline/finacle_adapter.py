"""
Core banking (Finacle-style) data adapter.

Why this exists: SACHET currently reads one static competition CSV. A real
deployment needs to pull account/transaction data from Bank of India's core
banking system — almost certainly Infosys Finacle, the dominant CBS across
Indian public-sector banks. Finacle integration in practice happens one of
two ways:

1. Batch file exchange (SFTP/managed-file-transfer) — the most common,
   universally-supported pattern for legacy/on-prem CBS deployments, where
   the bank exports a delimited file on a schedule (nightly or intraday).
2. A near-real-time API/BIP (Business Integration Platform) layer, where
   Finacle exposes events or callable services — faster, but requires a
   specific integration agreement with the bank's Finacle team that a
   hackathon has no way to verify or assume.

This module defines the adapter CONTRACT first (CoreBankingAdapter), so the
rest of the pipeline never depends on which integration method is actually
available, then provides two implementations: a batch-file adapter (the
safe default assumption) and a mock adapter (for local development/demo
without real bank data). Swapping in a real Finacle API adapter later means
implementing this same interface, not rewriting the pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import csv
import io
import random


@dataclass
class TransactionEvent:
    """One transaction, normalized to the shape the rest of the pipeline
    expects, regardless of which core-banking adapter produced it."""
    account_id: str
    timestamp: datetime
    channel: str          # e.g. CASH, CHEQUE, UPI, NEFT, IMPS, RTGS, ATM, POS,
                            # STDNG_INSTR, APB, BBPS, MBNKING, NET_BNKING, GST, LOAN, FEES_CHRGS
    direction: str         # CREDIT or DEBIT
    amount: float
    balance_after: Optional[float] = None
    is_customer_induced: bool = True
    # Whether the customer directly initiated this transaction (a UPI
    # payment they made) vs. a system/bank-induced posting (an interest
    # credit, a fee auto-debit). Default True because the overwhelming
    # majority of events on any of these channels are customer-initiated;
    # only feature_store.py's CI_-prefixed features (see schema_mapper.py)
    # actually filter on this flag.


class CoreBankingAdapter(ABC):
    """Contract every core-banking integration must satisfy. Nothing else in
    the pipeline should import a specific adapter directly — depend on this
    interface so the batch-vs-API decision can change without touching
    downstream code."""

    @abstractmethod
    def fetch_new_transactions(self, since: datetime) -> list[TransactionEvent]:
        """All transactions recorded since `since`, across all accounts."""
        raise NotImplementedError

    @abstractmethod
    def fetch_account_metadata(self, account_id: str) -> dict:
        """Static/slow-changing account context: product type, occupation,
        tenure, age — the fields SACHET uses for context and typology
        features, not raw transaction stats."""
        raise NotImplementedError


class FinacleBatchFileAdapter(CoreBankingAdapter):
    """Reads a Finacle-style delimited batch export. Field names below are a
    best-effort mapping based on how Finacle typically exports transaction
    data (documented assumption, not verified against a live Finacle
    instance — flag for confirmation with BOI's Finacle integration team
    before production use). Expected columns:
    ACCT_ID, TXN_DATE, TXN_TIME, CHANNEL, DR_CR, AMOUNT, BALANCE
    """

    def __init__(self, batch_directory: str):
        self.batch_directory = batch_directory

    def _parse_file(self, file_path: str) -> list[TransactionEvent]:
        events = []
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            required = {'ACCT_ID', 'TXN_DATE', 'TXN_TIME', 'CHANNEL', 'DR_CR', 'AMOUNT'}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"batch file {file_path} is missing required columns; "
                    f"expected {sorted(required)}, got {reader.fieldnames}"
                )
            for row in reader:
                ts = datetime.strptime(
                    f"{row['TXN_DATE']} {row['TXN_TIME']}", "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc)
                events.append(TransactionEvent(
                    account_id=row['ACCT_ID'],
                    timestamp=ts,
                    channel=row['CHANNEL'],
                    direction='CREDIT' if row['DR_CR'].upper() in ('CR', 'CREDIT') else 'DEBIT',
                    amount=float(row['AMOUNT']),
                    balance_after=float(row['BALANCE']) if row.get('BALANCE') else None,
                ))
        return events

    def fetch_new_transactions(self, since: datetime) -> list[TransactionEvent]:
        import os
        events = []
        for fname in sorted(os.listdir(self.batch_directory)):
            if fname.endswith('.csv'):
                events.extend(self._parse_file(os.path.join(self.batch_directory, fname)))
        return [e for e in events if e.timestamp >= since]

    def fetch_account_metadata(self, account_id: str) -> dict:
        raise NotImplementedError(
            "account metadata is expected via a SEPARATE Finacle export "
            "(customer master file), not the transaction batch — wire this "
            "once the actual export format is confirmed with BOI"
        )


class MockCoreBankingAdapter(CoreBankingAdapter):
    """Generates synthetic transaction events for local development and
    demos, with no dependency on real bank data or a live Finacle
    connection. NOT for production use — exists purely so the rest of the
    pipeline (feature store, event bus, scoring) can be developed and tested
    end-to-end before real CBS access is available."""

    _CHANNELS = ['CASH', 'CHEQUE', 'UPI', 'NEFT', 'IMPS', 'RTGS', 'ATM', 'POS']

    def __init__(self, n_accounts: int = 50, seed: int = 42):
        self._rng = random.Random(seed)
        self._accounts = [f"MOCK{i:06d}" for i in range(n_accounts)]

    def fetch_new_transactions(self, since: datetime) -> list[TransactionEvent]:
        n = self._rng.randint(5, 30)
        events = []
        for _ in range(n):
            events.append(TransactionEvent(
                account_id=self._rng.choice(self._accounts),
                timestamp=datetime.now(timezone.utc),
                channel=self._rng.choice(self._CHANNELS),
                direction=self._rng.choice(['CREDIT', 'DEBIT']),
                amount=round(self._rng.uniform(100, 500000), 2),
            ))
        return events

    def fetch_account_metadata(self, account_id: str) -> dict:
        return {
            'account_id': account_id,
            'product': self._rng.choice(['Savings', 'Current']),
            'occupation': self._rng.choice(['salaried', 'selfemployed', 'student']),
            'tenure_days': self._rng.randint(1, 3650),
        }
