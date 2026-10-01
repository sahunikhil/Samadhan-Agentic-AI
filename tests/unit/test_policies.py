"""The policy engine decides money and eligibility, so it gets exhaustive, table-driven tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from caseflow.domain.catalog import PRODUCTS
from caseflow.domain.policies import business_days_between, can_cancel, evaluate_return, is_shipment_stalled

NOW = datetime(2026, 9, 30, 12, 0)


def _ret(sku: str, *, tier: str = "standard", days: int = 10, opened: bool = True, seal: bool = True,
         reason: str = "changed_mind", status: str = "delivered", price: float | None = None):  # fmt: skip
    p = PRODUCTS[sku]
    return evaluate_return(
        product=p, quantity=1, unit_price=price or p.price, tier=tier, order_status=status,  # type: ignore[arg-type]
        delivered_at=NOW - timedelta(days=days), opened=opened, seal_intact=seal, reason=reason, now=NOW,  # type: ignore[arg-type]
    )  # fmt: skip


def test_opened_laptop_standard_customer_pays_restocking_and_shipping() -> None:
    d = _ret("VB-PRO-16")
    assert d.eligible
    assert d.restocking_fee == pytest.approx(284.85)
    assert d.return_shipping_fee == 6.99
    assert d.refund_amount == pytest.approx(1607.16)
    assert d.requires_human_approval


def test_plus_members_pay_no_fees_and_get_60_days() -> None:
    d = _ret("VB-AIR-14", tier="plus", days=45)
    assert d.eligible and d.window_days == 60
    assert d.restocking_fee == 0 and d.return_shipping_fee == 0
    assert d.refund_amount == 1099.00


def test_standard_window_is_30_days() -> None:
    d = _ret("AURA-SPK", days=45, opened=False)
    assert not d.eligible
    assert "30-day return window" in d.reasons[0]


def test_defective_item_outside_window_suggests_warranty() -> None:
    d = _ret("AURA-SPK", days=45, reason="defective")
    assert not d.eligible and d.suggested_alternative == "warranty_claim"


def test_earbuds_with_broken_seal_are_not_returnable_unless_defective() -> None:
    assert not _ret("PULSE-BUDS", seal=False).eligible
    assert _ret("PULSE-BUDS", seal=False, reason="defective").eligible
    assert _ret("PULSE-BUDS", opened=False, seal=True).eligible


def test_damaged_on_arrival_within_7_days_waives_all_fees() -> None:
    d = _ret("VB-AIR-14", days=3, reason="damaged_on_arrival")
    assert d.eligible and d.restocking_fee == 0 and d.return_shipping_fee == 0
    assert d.refund_amount == 1099.00


def test_damage_reported_late_is_treated_as_normal_return() -> None:
    d = _ret("NOVA-TAB-11", days=12, reason="damaged_on_arrival")
    assert d.eligible and d.restocking_fee > 0


def test_gift_cards_are_never_returnable() -> None:
    assert not _ret("GIFT-50", days=1).eligible


def test_processing_orders_should_be_cancelled_not_returned() -> None:
    d = _ret("NOVA-8-256", status="processing")
    assert not d.eligible and d.suggested_alternative == "cancel_order"


def test_store_credit_includes_10_percent_bonus() -> None:
    d = _ret("PULSE-ANC", opened=False, tier="plus")
    assert d.store_credit_amount == pytest.approx(249.00 * 1.10)


@pytest.mark.parametrize(("amount_sku", "needs_human"), [("PULSE-CUSH", False), ("PULSE-ANC", True)])
def test_auto_approve_limit(amount_sku: str, needs_human: bool) -> None:
    assert _ret(amount_sku, tier="plus").requires_human_approval is needs_human


def test_holiday_purchases_extend_to_january_31() -> None:
    delivered = datetime(2025, 12, 10)
    d = evaluate_return(
        product=PRODUCTS["AURA-SPK"], quantity=1, unit_price=129, tier="standard", order_status="delivered",
        delivered_at=delivered, opened=False, seal_intact=True, reason="changed_mind", now=datetime(2026, 1, 25),
    )  # fmt: skip
    assert d.eligible


def test_business_days_and_stalled_shipments() -> None:
    friday = datetime(2026, 9, 25)
    assert business_days_between(friday.date(), datetime(2026, 9, 28).date()) == 1  # over the weekend
    assert is_shipment_stalled(NOW - timedelta(days=8), "in_transit", NOW)
    assert not is_shipment_stalled(NOW - timedelta(days=2), "in_transit", NOW)
    assert not is_shipment_stalled(NOW - timedelta(days=30), "delivered", NOW)


def test_only_processing_orders_can_be_cancelled() -> None:
    assert can_cancel("processing")
    assert not any(can_cancel(s) for s in ("shipped", "in_transit", "delivered", "cancelled"))
