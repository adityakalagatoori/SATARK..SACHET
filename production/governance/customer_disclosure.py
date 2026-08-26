"""
Customer disclosure for AI-driven decisions.

Why this exists: RBI's draft Model Risk Management guidance requires
customer disclosure whenever an AI-driven decision affects them — a customer
whose transaction was auto-held has a right to know that happened and why,
in plain language, without exposing the model's internal logic (doing so
would help a genuine fraudster learn how to evade detection, which is a
real and separate risk to manage alongside disclosure).

Design choice — category-level reasoning, not feature-level: a customer
notice says "unusual account activity" or "a routine security check," never
"your SHAP score on cash-deposit velocity exceeded 0.7." Feature-level detail
belongs in the investigator-facing explainability store (see
explainability_store.py), which is internal, not customer-facing.
"""

from datetime import datetime, timezone

_TEMPLATES = {
    'AUTO-HOLD': (
        "We've placed a temporary hold on a recent transaction from your account "
        "as part of a routine security check. This is a precaution, not an "
        "accusation — most holds are cleared within a few hours once verified. "
        "No action is needed from you unless we contact you directly. "
        "If you believe this hold is a mistake, please contact Bank of India "
        "customer support with reference ID {reference_id}."
    ),
    'PRIORITY_REVIEW': (
        "Your account has been selected for a routine review as part of our "
        "ongoing account security process. This does not restrict your account "
        "in any way. Reference ID: {reference_id}."
    ),
    'STANDARD_QUEUE': (
        "As part of routine account monitoring, your account is included in a "
        "periodic security review. No action is required. Reference ID: {reference_id}."
    ),
}


def generate_disclosure_notice(account_id: int, action_type: str, reference_id: str) -> dict:
    """Build a plain-language, customer-facing disclosure for one automated
    action. Returns a dict ready to hand to an SMS/email/notification
    service — this module does not send anything itself, it only prepares
    the disclosed content, keeping the actual delivery channel a separate,
    swappable concern (Phase 2/3 integration)."""
    template = _TEMPLATES.get(action_type)
    if template is None:
        raise ValueError(
            f"no disclosure template for action_type={action_type!r}; "
            f"WATCHLIST-tier actions are not customer-facing by design "
            f"(they involve no restriction on the account)"
        )
    message = template.format(reference_id=reference_id)
    return {
        'account_id': account_id,
        'action_type': action_type,
        'reference_id': reference_id,
        'message': message,
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
