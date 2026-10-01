"""Deterministic demo data.

Dates are generated *relative to now* so return windows and "stuck parcel" rules
behave the same whenever the demo is run. Each order exists to exercise one
concrete support scenario (see ``SCENARIOS``), and the evaluation dataset
(``evals/datasets/agent_scenarios.jsonl``) is written against these IDs.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import delete

from caseflow.domain.catalog import PRODUCTS
from caseflow.mcp_servers.commerce.models import (
    CatalogPrice,
    CommerceBase,
    Customer,
    Order,
    OrderItem,
    Refund,
    ReturnRequest,
    Shipment,
)
from caseflow.mcp_servers.db import Database, utcnow

SCENARIOS: dict[str, str] = {
    "VW-10001": "Plus member, opened laptop delivered 12 days ago -> returnable, no restocking fee, refund > $100.",
    "VW-10002": "Earbuds with broken hygiene seal -> not returnable unless defective.",
    "VW-10003": "Charger + power bank in transit, on time.",
    "VW-10004": "Headphones return received by warehouse -> refund $242.01 pending -> needs human approval.",
    "VW-10005": "Phone order still processing -> can be cancelled or address changed.",
    "VW-10006": "Speaker delivered 45 days ago (standard) -> outside 30-day window.",
    "VW-10007": "Opened VoltBook Pro 16 (standard) -> 15% restocking fee + $6.99 shipping.",
    "VW-10008": "Power bank stuck in transit for 5+ business days -> carrier trace.",
    "VW-10009": "Plus member, speaker delivered 50 days ago -> within 60-day Plus window.",
    "VW-10010": "Charger price dropped $20 within 14 days -> price adjustment (auto-approved).",
    "VW-10011": "UK customer, sealed earbuds -> returnable.",
    "VW-10012": "Gift card -> non-returnable.",
    "VW-10013": "Ear cushions damaged on arrival -> instant refund $29 (auto-approved).",
    "VW-10014": "Dock order cancelled before shipping.",
    "VW-10015": "Tablet already refunded.",
}

CUSTOMERS = [
    ("cust_001", "Maya Chen", "maya.chen@example.com", "plus", "US", "1 Market St, San Francisco, CA 94105", 2480, 0.0),
    ("cust_002", "Daniel Okafor", "daniel.okafor@example.com", "standard", "US", "250 W 55th St, New York, NY 10019", 640, 0.0),
    ("cust_003", "Sofia Martins", "sofia.martins@example.com", "standard", "US", "742 Evergreen Ave, Austin, TX 78701", 1210, 0.0),
    ("cust_004", "Arjun Patel", "arjun.patel@example.com", "plus", "US", "88 Lake Shore Dr, Chicago, IL 60611", 5300, 25.0),
    ("cust_005", "Emma Johansson", "emma.johansson@example.com", "standard", "GB", "10 Baker St, London W1U 3BW, UK", 180, 0.0),
    ("cust_006", "Liam O'Brien", "liam.obrien@example.com", "standard", "US", "15 Pike St, Seattle, WA 98101", 90, 0.0),
]  # fmt: skip


@dataclass
class _OrderSpec:
    order_id: str
    customer_id: str
    items: list[tuple[str, int, float, bool, bool]]  # sku, qty, unit_price, opened, seal_intact
    status: str
    ordered_days_ago: float
    shipped_days_ago: float | None = None
    delivered_days_ago: float | None = None
    method: str = "standard"
    payment: str = "Visa ending 4242"
    carrier: str = "UPS"
    stalled_days_ago: float | None = None
    extra: dict[str, object] = field(default_factory=dict)


ORDERS: list[_OrderSpec] = [
    _OrderSpec("VW-10001", "cust_001", [("VB-AIR-14", 1, 1099.00, True, True)], "delivered", 16, 15, 12, method="express"),
    _OrderSpec("VW-10002", "cust_001", [("PULSE-BUDS", 1, 179.00, True, False)], "delivered", 9, 8, 5),
    _OrderSpec("VW-10003", "cust_001", [("VC-65W", 1, 49.00, False, True), ("PC-20K", 1, 59.00, False, True)], "in_transit", 3, 2, None, carrier="FedEx"),
    _OrderSpec("VW-10004", "cust_002", [("PULSE-ANC", 1, 249.00, True, True)], "return_requested", 26, 24, 20, payment="Mastercard ending 5454"),
    _OrderSpec("VW-10005", "cust_002", [("NOVA-8-256", 1, 799.00, False, True)], "processing", 0.1, payment="PayPal"),
    _OrderSpec("VW-10006", "cust_002", [("AURA-SPK", 1, 129.00, True, True)], "delivered", 50, 48, 45, payment="Mastercard ending 5454"),
    _OrderSpec("VW-10007", "cust_003", [("VB-PRO-16", 1, 1899.00, True, True)], "delivered", 14, 13, 10, payment="Amex ending 1005"),
    _OrderSpec("VW-10008", "cust_003", [("PC-20K", 1, 59.00, False, True)], "in_transit", 12, 11, None, carrier="USPS", stalled_days_ago=8, payment="Amex ending 1005"),
    _OrderSpec("VW-10009", "cust_004", [("AURA-SPK", 1, 129.00, True, True)], "delivered", 54, 53, 50, method="express"),
    _OrderSpec("VW-10010", "cust_004", [("VC-140W", 1, 89.00, True, True)], "delivered", 6, 5, 3, method="express"),
    _OrderSpec("VW-10011", "cust_005", [("PULSE-BUDS", 1, 179.00, False, True)], "delivered", 13, 12, 4, method="international", carrier="DHL", payment="Visa ending 1881"),
    _OrderSpec("VW-10012", "cust_005", [("GIFT-50", 1, 50.00, True, True)], "delivered", 22, 21, 20, method="digital", carrier="Email", payment="Visa ending 1881"),
    _OrderSpec("VW-10013", "cust_006", [("PULSE-CUSH", 1, 29.00, True, True)], "delivered", 6, 5, 2, payment="Google Pay"),
    _OrderSpec("VW-10014", "cust_006", [("VOLT-DOCK", 1, 199.00, False, True)], "cancelled", 30, payment="Google Pay"),
    _OrderSpec("VW-10015", "cust_006", [("NOVA-TAB-11", 1, 449.00, True, True)], "refunded", 80, 78, 70, payment="Google Pay"),
]  # fmt: skip

PROMO_PRICES = {"VC-140W": 69.00}  # current promotion -> enables the price-adjustment scenario


def _tracking_events(spec: _OrderSpec, now: datetime) -> tuple[str, list[dict[str, str]], datetime | None]:
    if spec.shipped_days_ago is None:
        return "label_pending", [], None
    shipped = now - timedelta(days=spec.shipped_days_ago)
    hubs = ["Reno, NV", "Salt Lake City, UT", "Denver, CO", "Kansas City, MO", "Chicago, IL"]
    events = [
        {
            "at": (shipped - timedelta(hours=6)).isoformat(),
            "location": "Reno, NV",
            "status": "label_created",
            "description": "Shipping label created",
        }
    ]
    events.append(
        {
            "at": shipped.isoformat(),
            "location": "Reno, NV",
            "status": "picked_up",
            "description": "Picked up by carrier",
        }
    )

    if spec.delivered_days_ago is not None:
        delivered = now - timedelta(days=spec.delivered_days_ago)
        span = (delivered - shipped) / 3
        for i, hub in enumerate(hubs[1:3], start=1):
            events.append(
                {
                    "at": (shipped + span * i).isoformat(),
                    "location": hub,
                    "status": "in_transit",
                    "description": "Arrived at carrier facility",
                }
            )
        events.append(
            {
                "at": (delivered - timedelta(hours=5)).isoformat(),
                "location": "Local facility",
                "status": "out_for_delivery",
                "description": "Out for delivery",
            }
        )
        events.append(
            {"at": delivered.isoformat(), "location": "Front door", "status": "delivered", "description": "Delivered"}
        )
        return "delivered", events, delivered

    last = now - timedelta(days=spec.stalled_days_ago) if spec.stalled_days_ago else now - timedelta(hours=10)
    events.append(
        {
            "at": last.isoformat(),
            "location": "Denver, CO",
            "status": "in_transit",
            "description": "Departed carrier facility",
        }
    )
    eta = shipped + timedelta(days=5) if spec.stalled_days_ago else now + timedelta(days=2)
    return "in_transit", events, eta


async def seed_commerce(db: Database, *, reset: bool = True) -> dict[str, int]:
    await db.create_all(CommerceBase)
    now = utcnow()
    async with db.session() as s:
        if reset:
            for model in (Refund, ReturnRequest, Shipment, OrderItem, Order, CatalogPrice, Customer):
                await s.execute(delete(model))

        for cid, name, email, tier, country, address, points, credit in CUSTOMERS:
            s.add(Customer(id=cid, name=name, email=email, tier=tier, country=country, default_address=address,
                           voltpoints=points, store_credit=credit, member_since=now - timedelta(days=400)))  # fmt: skip
        for sku, product in PRODUCTS.items():
            s.add(CatalogPrice(sku=sku, price=PROMO_PRICES.get(sku, product.price), stock=250))
        await s.flush()

        tiers = {c[0]: c[3] for c in CUSTOMERS}
        addresses = {c[0]: c[5] for c in CUSTOMERS}
        for spec in ORDERS:
            subtotal = round(sum(q * p for _, q, p, _, _ in spec.items), 2)
            if spec.method == "express":
                shipping = 0.0 if tiers[spec.customer_id] == "plus" else 14.99
            elif spec.method in ("international",):
                shipping = 19.99 if subtotal < 150 else 0.0
            elif spec.method == "digital":
                shipping = 0.0
            else:
                shipping = 0.0 if subtotal >= 50 else 5.99
            tax = 0.0 if spec.method in ("international", "digital") else round(subtotal * 0.08, 2)
            ship_status, events, eta_or_delivered = _tracking_events(spec, now)
            order = Order(
                id=spec.order_id,
                customer_id=spec.customer_id,
                status=spec.status,
                created_at=now - timedelta(days=spec.ordered_days_ago),
                shipped_at=now - timedelta(days=spec.shipped_days_ago) if spec.shipped_days_ago is not None else None,
                delivered_at=now - timedelta(days=spec.delivered_days_ago)
                if spec.delivered_days_ago is not None
                else None,
                shipping_method=spec.method,
                shipping_address=addresses[spec.customer_id],
                subtotal=subtotal,
                shipping_cost=shipping,
                tax=tax,
                total=round(subtotal + shipping + tax, 2),
                payment_method=spec.payment,
            )
            s.add(order)
            for sku, qty, price, opened, seal in spec.items:
                s.add(OrderItem(order_id=spec.order_id, sku=sku, name=PRODUCTS[sku].name, quantity=qty,
                                unit_price=price, opened=opened, seal_intact=seal))  # fmt: skip
            if spec.shipped_days_ago is not None and spec.status != "cancelled":
                s.add(Shipment(
                    order_id=spec.order_id, carrier=spec.carrier,
                    tracking_number=f"{spec.carrier[:2].upper()}{zlib.crc32(spec.order_id.encode()):010d}",
                    status=ship_status, events=events,
                    estimated_delivery=eta_or_delivered,
                ))  # fmt: skip

        # Pre-existing after-sales records that make scenarios realistic.
        s.add(ReturnRequest(id="RMA-20001", order_id="VW-10004", customer_id="cust_002", sku="PULSE-ANC",
                            reason="changed_mind", resolution="refund", status="received", refund_amount=242.01,
                            return_shipping_fee=6.99, created_at=now - timedelta(days=9),
                            label_url="https://returns.voltwise.example/labels/RMA-20001.pdf"))  # fmt: skip
        s.add(Refund(id="RF-30001", order_id="VW-10015", customer_id="cust_006", amount=449.00 * 1.08,
                     method="original_payment", reason="return_received", status="processed",
                     approved_by="agent:jlee", created_at=now - timedelta(days=60)))  # fmt: skip
    return {"customers": len(CUSTOMERS), "orders": len(ORDERS)}
