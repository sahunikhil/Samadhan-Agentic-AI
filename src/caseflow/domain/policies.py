"""Deterministic policy engine.

The LLM decides *what the customer wants*; this module decides *what policy
allows*. Money and eligibility are never left to a language model: these rules
are unit-tested, versioned code, and the MCP server enforces them server-side no
matter what arguments the model sends.

Every rule references the knowledge-base article that documents it, so the
agent's grounded explanations and the system's actual behavior cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Literal

from caseflow.domain.catalog import Product

Tier = Literal["standard", "plus"]
ReturnReason = Literal["changed_mind", "defective", "wrong_item", "damaged_on_arrival", "not_as_described", "other"]

STANDARD_RETURN_DAYS = 30  # KB-003
PLUS_RETURN_DAYS = 60  # KB-003, KB-010
RESTOCKING_FEE_RATE = 0.15  # KB-003: opened laptops/tablets
RETURN_SHIPPING_FEE = 6.99  # KB-003
STORE_CREDIT_BONUS = 0.10  # KB-003, KB-011
DAMAGE_REPORT_DAYS = 7  # KB-004
STALLED_SHIPMENT_BUSINESS_DAYS = 5  # KB-001
AUTO_APPROVE_REFUND_LIMIT = 100.0  # KB-003 "Refund approvals"


@dataclass(slots=True)
class ReturnDecision:
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    window_days: int = STANDARD_RETURN_DAYS
    days_since_delivery: int | None = None
    item_value: float = 0.0
    restocking_fee: float = 0.0
    return_shipping_fee: float = 0.0
    refund_amount: float = 0.0
    store_credit_amount: float = 0.0
    requires_human_approval: bool = False
    suggested_alternative: str | None = None
    policy_refs: list[str] = field(default_factory=lambda: ["KB-003"])


def _holiday_extended_deadline(delivered: date) -> date | None:
    """Holiday purchases (Nov 1 - Dec 24) are returnable until Jan 31 next year (KB-003)."""
    if (delivered.month == 11) or (delivered.month == 12 and delivered.day <= 24):
        return date(delivered.year + 1, 1, 31)
    return None


def evaluate_return(
    *,
    product: Product,
    quantity: int,
    unit_price: float,
    tier: Tier,
    order_status: str,
    delivered_at: datetime | None,
    opened: bool,
    seal_intact: bool,
    reason: ReturnReason,
    now: datetime,
    auto_approve_limit: float = AUTO_APPROVE_REFUND_LIMIT,
) -> ReturnDecision:
    window = PLUS_RETURN_DAYS if tier == "plus" else STANDARD_RETURN_DAYS
    decision = ReturnDecision(eligible=False, window_days=window, item_value=round(unit_price * quantity, 2))

    if order_status in {"processing"}:
        decision.reasons.append("The order has not shipped yet, so it can simply be cancelled instead of returned.")
        decision.suggested_alternative = "cancel_order"
        decision.policy_refs = ["KB-009"]
        return decision
    if delivered_at is None or order_status in {"shipped", "in_transit", "out_for_delivery"}:
        decision.reasons.append("The order has not been delivered yet; returns start after delivery.")
        decision.policy_refs = ["KB-009", "KB-003"]
        return decision
    if order_status in {"cancelled", "refunded"}:
        decision.reasons.append(f"The order is already {order_status}.")
        return decision
    if not product.returnable:
        decision.reasons.append(f"{product.name} is non-returnable (gift cards, software and Final Sale items).")
        decision.policy_refs = ["KB-003", "KB-011"]
        return decision

    days = (now.date() - delivered_at.date()).days
    decision.days_since_delivery = days
    damage_reported_in_time = reason == "damaged_on_arrival" and days <= DAMAGE_REPORT_DAYS
    fault_is_ours = reason in {"defective", "wrong_item"} or damage_reported_in_time

    deadline = delivered_at.date() + timedelta(days=window)
    if holiday_deadline := _holiday_extended_deadline(delivered_at.date()):
        deadline = max(deadline, holiday_deadline)
    if now.date() > deadline:
        decision.reasons.append(
            f"The {window}-day return window ended on {deadline.isoformat()} ({days} days since delivery)."
        )
        if reason == "defective" and days <= product.warranty_months * 30:
            decision.suggested_alternative = "warranty_claim"
            decision.reasons.append("The product is still under warranty, so a warranty repair/replacement applies.")
            decision.policy_refs = ["KB-003", "KB-005"]
        return decision

    if product.hygiene_sensitive and opened and not seal_intact and not fault_is_ours:
        decision.reasons.append(
            "In-ear products can only be returned with the hygiene seal intact unless they are defective."
        )
        decision.policy_refs = ["KB-003", "KB-017"]
        return decision

    decision.eligible = True
    if product.restocking_fee_applies and opened and not fault_is_ours and tier != "plus":
        decision.restocking_fee = round(decision.item_value * RESTOCKING_FEE_RATE, 2)
        decision.reasons.append("A 15% restocking fee applies to opened laptops and tablets.")
    if not fault_is_ours and tier != "plus":
        decision.return_shipping_fee = RETURN_SHIPPING_FEE
        decision.reasons.append("$6.99 return shipping is deducted because the item is not defective.")
    if fault_is_ours:
        decision.reasons.append("No fees apply because the item is defective, damaged or incorrect.")
        decision.policy_refs = ["KB-003", "KB-004"]
    if tier == "plus":
        decision.reasons.append("VoltPlus: no restocking fee and free return shipping.")
        decision.policy_refs = [*decision.policy_refs, "KB-010"]

    decision.refund_amount = round(
        max(decision.item_value - decision.restocking_fee - decision.return_shipping_fee, 0.0), 2
    )
    decision.store_credit_amount = round(decision.refund_amount * (1 + STORE_CREDIT_BONUS), 2)
    decision.requires_human_approval = decision.refund_amount > auto_approve_limit
    return decision


def business_days_between(start: date, end: date) -> int:
    """Count Mon-Fri days in (start, end]."""
    if end <= start:
        return 0
    days, current = 0, start
    while current < end:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days += 1
    return days


def is_shipment_stalled(last_event_at: datetime, status: str, now: datetime) -> bool:
    """No tracking movement for 5+ business days on an undelivered parcel (KB-001)."""
    if status in {"delivered", "cancelled"}:
        return False
    return business_days_between(last_event_at.date(), now.date()) >= STALLED_SHIPMENT_BUSINESS_DAYS


def can_cancel(order_status: str) -> bool:
    """Orders can be cancelled only before they ship (KB-009)."""
    return order_status == "processing"
