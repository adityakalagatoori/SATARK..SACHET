"""Regression tests for the Phase 1 RBI-compliance governance layer."""

from datetime import datetime, timezone, timedelta

import pytest
import kill_switch
import human_override
import customer_disclosure
import encryption


def test_kill_switch_starts_inactive():
    assert kill_switch.is_active() is False


def test_kill_switch_activate_deactivate_cycle():
    kill_switch.activate(triggered_by='admin_1', reason='drill')
    assert kill_switch.is_active() is True
    kill_switch.deactivate(triggered_by='admin_1', reason='drill complete')
    assert kill_switch.is_active() is False


def test_kill_switch_requires_reason():
    with pytest.raises(ValueError):
        kill_switch.activate(triggered_by='admin_1', reason='')
    with pytest.raises(ValueError):
        kill_switch.activate(triggered_by='', reason='no actor')


def test_kill_switch_history_is_append_only_and_ordered():
    kill_switch.activate(triggered_by='a', reason='r1')
    kill_switch.deactivate(triggered_by='a', reason='r2')
    kill_switch.activate(triggered_by='a', reason='r3')
    history = kill_switch.get_history()
    assert len(history) == 3
    assert [h['reason'] for h in history] == ['r3', 'r2', 'r1']  # most recent first


def test_human_override_action_lifecycle():
    action_id = human_override.record_action(
        account_id=42, action_type='AUTO-HOLD', action_detail='test hold', model_version='v1'
    )
    status = human_override.get_action_status(42)
    assert len(status) == 1
    assert status[0]['status'] == 'ACTIVE'

    human_override.override_action(action_id, investigator_id='inv_1', reason='verified legitimate')
    status = human_override.get_action_status(42)
    assert status[0]['status'] == 'OVERRIDDEN'
    assert status[0]['override']['investigator_id'] == 'inv_1'


def test_human_override_original_record_never_deleted():
    """The append-only guarantee: overriding must not remove the original
    action record, only supersede its status."""
    action_id = human_override.record_action(account_id=7, action_type='AUTO-HOLD')
    human_override.override_action(action_id, investigator_id='inv_1', reason='cleared')
    status = human_override.get_action_status(7)
    assert len(status) == 1  # still exactly one record, not deleted, not duplicated
    assert status[0]['id'] == action_id


def test_human_override_cannot_double_override():
    action_id = human_override.record_action(account_id=7, action_type='AUTO-HOLD')
    human_override.override_action(action_id, investigator_id='inv_1', reason='first')
    with pytest.raises(ValueError):
        human_override.override_action(action_id, investigator_id='inv_2', reason='second')


def test_human_override_requires_investigator_and_reason():
    action_id = human_override.record_action(account_id=7, action_type='AUTO-HOLD')
    with pytest.raises(ValueError):
        human_override.override_action(action_id, investigator_id='', reason='missing actor')


def test_record_action_default_sla_sets_expiry_roughly_48h_out():
    action_id = human_override.record_action(account_id=100, action_type='AUTO-HOLD')
    status = human_override.get_action_status(100)
    action = [a for a in status if a['id'] == action_id][0]
    assert action['expires_at'] is not None
    expires_at = datetime.fromisoformat(action['expires_at'])
    created_at = datetime.fromisoformat(action['created_at'])
    delta_hours = (expires_at - created_at).total_seconds() / 3600
    assert 47.9 < delta_hours < 48.1


def test_record_action_custom_sla_is_respected():
    action_id = human_override.record_action(
        account_id=101, action_type='AUTO-HOLD', sla_hours=6.0
    )
    status = human_override.get_action_status(101)
    action = [a for a in status if a['id'] == action_id][0]
    expires_at = datetime.fromisoformat(action['expires_at'])
    created_at = datetime.fromisoformat(action['created_at'])
    delta_hours = (expires_at - created_at).total_seconds() / 3600
    assert 5.9 < delta_hours < 6.1


def test_auto_expire_stale_actions_expires_past_due_and_skips_future():
    stale_id = human_override.record_action(
        account_id=102, action_type='AUTO-HOLD', sla_hours=-1.0  # already expired
    )
    fresh_id = human_override.record_action(
        account_id=103, action_type='AUTO-HOLD', sla_hours=48.0  # far in the future
    )
    expired_ids = human_override.auto_expire_stale_actions()
    assert stale_id in expired_ids
    assert fresh_id not in expired_ids

    stale_status = human_override.get_action_status(102)[0]
    assert stale_status['status'] == 'OVERRIDDEN'

    fresh_status = human_override.get_action_status(103)[0]
    assert fresh_status['status'] == 'ACTIVE'


def test_auto_expired_action_is_distinguishable_from_human_override():
    auto_id = human_override.record_action(
        account_id=104, action_type='AUTO-HOLD', sla_hours=-1.0
    )
    human_id = human_override.record_action(
        account_id=105, action_type='AUTO-HOLD', sla_hours=48.0
    )
    human_override.override_action(human_id, investigator_id='inv_1', reason='verified legitimate')
    human_override.auto_expire_stale_actions()

    auto_status = human_override.get_action_status(104)[0]
    assert auto_status['override']['override_type'] == 'AUTO_EXPIRED'
    assert auto_status['override']['investigator_id'] == 'SYSTEM_AUTO_EXPIRY'

    human_status = human_override.get_action_status(105)[0]
    assert human_status['override']['override_type'] == 'HUMAN'
    assert human_status['override']['investigator_id'] == 'inv_1'


def test_auto_expire_skips_actions_already_overridden_by_human():
    action_id = human_override.record_action(
        account_id=106, action_type='AUTO-HOLD', sla_hours=-1.0  # would be stale
    )
    human_override.override_action(action_id, investigator_id='inv_1', reason='cleared before SLA check')
    expired_ids = human_override.auto_expire_stale_actions()
    assert action_id not in expired_ids  # already OVERRIDDEN, not ACTIVE — must not be double-processed

    status = human_override.get_action_status(106)[0]
    assert status['override']['override_type'] == 'HUMAN'
    assert status['override']['investigator_id'] == 'inv_1'


def test_record_action_backward_compatible_without_sla_hours():
    """Pre-existing call pattern: only the original required/optional args,
    no sla_hours — must still work exactly as before."""
    action_id = human_override.record_action(
        account_id=200, action_type='AUTO-HOLD', action_detail='legacy call', model_version='v1'
    )
    status = human_override.get_action_status(200)
    assert len(status) == 1
    assert status[0]['status'] == 'ACTIVE'
    assert status[0]['id'] == action_id


def test_customer_disclosure_generates_message_with_reference_id():
    notice = customer_disclosure.generate_disclosure_notice(
        account_id=9001, action_type='AUTO-HOLD', reference_id='REF-123'
    )
    assert 'REF-123' in notice['message']
    assert notice['action_type'] == 'AUTO-HOLD'


def test_customer_disclosure_rejects_non_customer_facing_watchlist_tier():
    """WATCHLIST-tier actions are internal-only by design (no restriction on
    the account), so no disclosure template should exist for them."""
    with pytest.raises(ValueError):
        customer_disclosure.generate_disclosure_notice(
            account_id=1, action_type='WATCHLIST', reference_id='REF-1'
        )


def test_encryption_round_trip():
    original = "investigator reason: verified legitimate business transfer"
    key = encryption.Fernet.generate_key()
    ct = encryption.encrypt_field(original, key=key)
    assert ct != original
    assert encryption.decrypt_field(ct, key=key) == original


def test_encryption_wrong_key_fails():
    key1 = encryption.Fernet.generate_key()
    key2 = encryption.Fernet.generate_key()
    ct = encryption.encrypt_field("secret", key=key1)
    with pytest.raises(Exception):
        encryption.decrypt_field(ct, key=key2)
