"""
Shared pytest fixtures. Every governance/auth module persists to a SQLite
file under its own package's _state/ directory — this fixture wipes those
directories before each test so tests never leak state into each other
(a test asserting "kill switch starts inactive" would be flaky otherwise,
since the switch's state is whatever the last test left it as).
"""

import sys
import os
import shutil
import pytest

_PROD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ('governance', 'feature_pipeline', 'serving', 'training', 'monitoring'):
    sys.path.insert(0, os.path.join(_PROD_ROOT, sub))


@pytest.fixture(autouse=True)
def clean_state():
    state_dirs = [
        os.path.join(_PROD_ROOT, 'governance', '_state'),
        os.path.join(_PROD_ROOT, 'serving', '_state'),
        os.path.join(_PROD_ROOT, 'training', '_registry'),
    ]
    for d in state_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
    yield
    for d in state_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
