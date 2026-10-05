"""Commerce MCP server - orders, shipments, returns and refunds.

MCP features demonstrated here
------------------------------
* **Per-customer auth**: ``customer_id = TokenClaim("sub")`` is injected from the
  verified JWT and is *invisible* to the model (not in the tool schema).
* **Scope-based authorization** per tool with ``auth=require_scopes(...)``.
* **Tool annotations** (``readOnlyHint`` / ``destructiveHint`` / ``idempotentHint``)
  that the agent side reads to decide which calls need human approval.
* **Structured output**: Pydantic return types become ``outputSchema`` +
  ``structuredContent``, surfaced to LangChain as a typed ``artifact``.
* **Modern stateless elicitation** (MCP 2026-07-28): ``cancel_order`` returns an
  ``InputRequiredResult`` asking the *customer* to confirm; LangChain turns it into a
  LangGraph ``interrupt()``.
* **ToolError** for business-rule violations the model can read and recover from.
* **Server-side enforcement** of money rules (policy engine + signed approvals) so
  the model's arguments are never trusted blindly.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import require_scopes
from fastmcp.server.dependencies import TokenClaim
from mcp.types import ElicitRequest, ElicitRequestFormParams, ElicitResult, InputRequiredResult, ToolAnnotations
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from starlette.requests import Request
from starlette.responses import JSONResponse

from samadhan.config import Settings, get_settings
from samadhan.domain.catalog import PRODUCTS, get_product
from samadhan.domain.policies import (
    AUTO_APPROVE_REFUND_LIMIT,
    ReturnReason,
    can_cancel,
    evaluate_return,
    is_shipment_stalled,
)
from samadhan.mcp_servers.auth import ApprovalError, build_verifier, verify_approval_code
from samadhan.mcp_servers.commerce.models import (
    CatalogPrice,
    CommerceBase,
    Customer,
    Order,
    Refund,
    ReturnRequest,
)
from samadhan.mcp_servers.db import Database, utcnow

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False, idempotentHint=True)
WRITE_SAFE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)


# ---- structured outputs (become MCP outputSchema) ---------------------------------------


class OrderLine(BaseModel):
    sku: str
    name: str
    quantity: int
    unit_price: float


class OrderSummary(BaseModel):
    order_id: str
    status: str
    ordered_at: str
    total: float
    items: list[OrderLine]
    delivered_at: str | None = None


class OrderDetail(OrderSummary):
    shipping_method: str
    shipping_address: str
    payment_method: str
    subtotal: float
    shipping_cost: float
    tax: float
    carrier: str | None = None
    tracking_number: str | None = None
    can_cancel: bool
    return_ids: list[str] = Field(default_factory=list)
    refunded_total: float = 0.0


class TrackingEvent(BaseModel):
    at: str
    location: str
    status: str
    description: str


class TrackingInfo(BaseModel):
    order_id: str
    carrier: str
    tracking_number: str
    status: str
    estimated_delivery: str | None
    stalled: bool
    advice: str
    events: list[TrackingEvent]


class ReturnEligibility(BaseModel):
    order_id: str
    sku: str
    eligible: bool
    reasons: list[str]
    window_days: int
    days_since_delivery: int | None
    item_value: float
    restocking_fee: float
    return_shipping_fee: float
    refund_amount: float
    store_credit_amount: float
    requires_human_approval: bool
    suggested_alternative: str | None
    policy_refs: list[str]
    next_step: str = ""


class ReturnCreated(BaseModel):
    rma_id: str
    order_id: str
    sku: str
    status: str
    resolution: str
    refund_amount: float
    instant_refund_eligible: bool
    label_url: str
    next_steps: str


class RefundReceipt(BaseModel):
    refund_id: str
    order_id: str
    amount: float
    method: str
    status: str
    approved_by: str
    expected_posting: str


class CustomerProfile(BaseModel):
    customer_id: str
    name: str
    tier: str
    country: str
    member_since: str
    voltpoints: int
    store_credit: float
    email_masked: str


class PriceAdjustmentCheck(BaseModel):
    order_id: str
    eligible: bool
    days_since_purchase: int
    lines: list[dict[str, Any]]
    already_refunded: float
    refundable_amount: float
    next_step: str


class ProductInfo(BaseModel):
    sku: str
    name: str
    category: str
    list_price: float
    current_price: float
    in_stock: bool
    warranty_months: int
    returnable: bool


# ---- server factory ---------------------------------------------------------------------


def create_commerce_server(settings: Settings | None = None, db: Database | None = None) -> FastMCP:
    settings = settings or get_settings()
    database = db or Database(settings.mcp.commerce_db_url)

    @asynccontextmanager
    async def lifespan(_: FastMCP) -> AsyncIterator[dict[str, Any]]:
        await database.create_all(CommerceBase)
        yield {"db": database}

    mcp = FastMCP(
        name="voltwise-commerce",
        instructions=(
            "Voltwise order management. All tools act on behalf of the authenticated customer; "
            "you can never access another customer's data. Always check return eligibility "
            "before creating a return, and never invent order IDs."
        ),
        auth=build_verifier(settings.mcp),
        lifespan=lifespan,
        mask_error_details=True,  # unexpected exceptions become generic errors; ToolError text is kept
    )

    async def _load_order(session: Any, order_id: str, customer_id: str) -> Order:
        order = await session.get(Order, order_id.strip().upper())
        # Same error for "doesn't exist" and "belongs to someone else": no ID enumeration.
        if order is None or order.customer_id != customer_id:
            raise ToolError(f"Order {order_id} was not found on this account.")
        return order

    def _summary(order: Order) -> dict[str, Any]:
        return {
            "order_id": order.id,
            "status": order.status,
            "ordered_at": order.created_at.date().isoformat(),
            "total": order.total,
            "delivered_at": order.delivered_at.date().isoformat() if order.delivered_at else None,
            "items": [
                OrderLine(sku=i.sku, name=i.name, quantity=i.quantity, unit_price=i.unit_price) for i in order.items
            ],
        }

    # ---- read tools -----------------------------------------------------------------

    @mcp.tool(annotations=READ_ONLY, auth=require_scopes("profile:read"), tags={"account"})
    async def get_customer_profile(customer_id: str = TokenClaim("sub")) -> CustomerProfile:
        """Get the signed-in customer's profile: name, membership tier (standard or VoltPlus), points and store credit."""
        async with database.session() as s:
            c = await s.get(Customer, customer_id)
            if c is None:
                raise ToolError("Customer profile not found.")
            user, _, domain = c.email.partition("@")
            return CustomerProfile(
                customer_id=c.id,
                name=c.name,
                tier=c.tier,
                country=c.country,
                member_since=c.member_since.date().isoformat(),
                voltpoints=c.voltpoints,
                store_credit=c.store_credit,
                email_masked=f"{user[:2]}***@{domain}",
            )

    @mcp.tool(annotations=READ_ONLY, auth=require_scopes("orders:read"), tags={"orders"})
    async def list_orders(
        status: str | None = None, limit: int = 10, customer_id: str = TokenClaim("sub")
    ) -> list[OrderSummary]:
        """List the customer's most recent orders, newest first. Optionally filter by status
        (processing, shipped, in_transit, delivered, cancelled, return_requested, refunded)."""
        async with database.session() as s:
            q = select(Order).where(Order.customer_id == customer_id).order_by(Order.created_at.desc())
            if status:
                q = q.where(Order.status == status)
            orders = (await s.scalars(q.limit(max(1, min(limit, 25))))).all()
            return [OrderSummary(**_summary(o)) for o in orders]

    @mcp.tool(annotations=READ_ONLY, auth=require_scopes("orders:read"), tags={"orders"})
    async def get_order(order_id: str, customer_id: str = TokenClaim("sub")) -> OrderDetail:
        """Get full details for one order: items, totals, shipping, payment, returns and refunds."""
        async with database.session() as s:
            order = await _load_order(s, order_id, customer_id)
            rmas = (await s.scalars(select(ReturnRequest.id).where(ReturnRequest.order_id == order.id))).all()
            refunded = await s.scalar(
                select(func.coalesce(func.sum(Refund.amount), 0.0)).where(Refund.order_id == order.id)
            )
            return OrderDetail(
                **_summary(order),
                shipping_method=order.shipping_method,
                shipping_address=order.shipping_address,
                payment_method=order.payment_method,
                subtotal=order.subtotal,
                shipping_cost=order.shipping_cost,
                tax=order.tax,
                carrier=order.shipment.carrier if order.shipment else None,
                tracking_number=order.shipment.tracking_number if order.shipment else None,
                can_cancel=can_cancel(order.status),
                return_ids=list(rmas),
                refunded_total=round(float(refunded or 0.0), 2),
            )

    @mcp.tool(annotations=READ_ONLY, auth=require_scopes("orders:read"), tags={"orders"})
    async def track_shipment(order_id: str, customer_id: str = TokenClaim("sub")) -> TrackingInfo:
        """Get live carrier tracking for an order, including whether the parcel is stalled (no movement for 5+ business days)."""
        async with database.session() as s:
            order = await _load_order(s, order_id, customer_id)
            if order.shipment is None:
                raise ToolError(f"Order {order.id} has not shipped yet (status: {order.status}).")
            sh = order.shipment
            events = [TrackingEvent(**e) for e in sh.events]
            last_at = datetime.fromisoformat(events[-1].at) if events else order.created_at
            stalled = is_shipment_stalled(last_at, sh.status, utcnow())
            if stalled:
                advice = (
                    "No carrier movement for 5+ business days. Per the Shipping Policy we can open a carrier trace; "
                    "if not found within 5 business days we reship or refund."
                )
            elif sh.status == "delivered":
                advice = "Delivered. If it is missing, wait 48 hours and check with neighbors before we open an investigation."
            else:
                advice = "On track."
            return TrackingInfo(
                order_id=order.id,
                carrier=sh.carrier,
                tracking_number=sh.tracking_number,
                status=sh.status,
                estimated_delivery=sh.estimated_delivery.date().isoformat() if sh.estimated_delivery else None,
                stalled=stalled,
                advice=advice,
                events=events,
            )

    @mcp.tool(annotations=READ_ONLY, tags={"catalog"})
    async def search_products(query: str = "", category: str | None = None) -> list[ProductInfo]:
        """Search the Voltwise catalog by name or SKU and get current price and stock."""
        async with database.session() as s:
            prices = {p.sku: p for p in (await s.scalars(select(CatalogPrice))).all()}
        q = query.lower().strip()
        results = []
        for sku, p in PRODUCTS.items():
            if category and p.category != category:
                continue
            if q and q not in p.name.lower() and q not in sku.lower():
                continue
            cp = prices.get(sku)
            results.append(ProductInfo(
                sku=sku, name=p.name, category=p.category, list_price=p.price,
                current_price=cp.price if cp else p.price, in_stock=bool(cp and cp.stock > 0),
                warranty_months=p.warranty_months, returnable=p.returnable,
            ))  # fmt: skip
        return results[:10]

    @mcp.tool(annotations=READ_ONLY, auth=require_scopes("orders:read"), tags={"returns"})
    async def check_return_eligibility(
        order_id: str,
        sku: str,
        reason: ReturnReason = "changed_mind",
        item_opened: bool | None = None,
        seal_intact: bool | None = None,
        customer_id: str = TokenClaim("sub"),
    ) -> ReturnEligibility:
        """Check whether an item can be returned and compute the exact refund, fees and store-credit option.

        Always call this before create_return. `reason` is one of: changed_mind, defective, wrong_item,
        damaged_on_arrival, not_as_described, other. Leave item_opened/seal_intact empty to use the
        condition recorded on the order."""
        async with database.session() as s:
            order = await _load_order(s, order_id, customer_id)
            customer = await s.get(Customer, customer_id)
            line = next((i for i in order.items if i.sku == sku.strip().upper()), None)
            product = get_product(sku)
            if line is None or product is None or customer is None:
                raise ToolError(f"SKU {sku} is not part of order {order.id}.")
            d = evaluate_return(
                product=product,
                quantity=line.quantity,
                unit_price=line.unit_price,
                tier=customer.tier,  # type: ignore[arg-type]
                order_status=order.status,
                delivered_at=order.delivered_at,
                opened=line.opened if item_opened is None else item_opened,
                seal_intact=line.seal_intact if seal_intact is None else seal_intact,
                reason=reason,
                now=utcnow(),
                auto_approve_limit=settings.agent.refund_auto_approve_limit,
            )
            next_step = (
                "Tell the customer the refund amount, fees and store-credit option. Only call create_return if "
                "they explicitly asked to START a return; if they asked whether they can or how much they would "
                "get, offer to start it instead - creating an RMA is an action, not an answer."
                if d.eligible
                else "Explain the reason above and offer the suggested alternative, if any. Do not create a return."
            )
            return ReturnEligibility(order_id=order.id, sku=line.sku, next_step=next_step, **asdict(d))

    # ---- write tools ----------------------------------------------------------------

    @mcp.tool(annotations=WRITE_SAFE, auth=require_scopes("returns:write"), tags={"returns"})
    async def create_return(
        order_id: str,
        sku: str,
        reason: ReturnReason,
        ctx: Context,
        resolution: Literal["refund", "store_credit", "exchange"] = "refund",
        item_opened: bool | None = None,
        seal_intact: bool | None = None,
        customer_id: str = TokenClaim("sub"),
    ) -> ReturnCreated | dict[str, Any] | InputRequiredResult:
        """Create a return (RMA) and email a prepaid label. The customer is asked to confirm the exact
        amounts first. Idempotent: calling it again for the same order and SKU returns the existing RMA.
        Fails if the item is not eligible."""
        async with database.session() as s:
            order = await _load_order(s, order_id, customer_id)
            existing = await s.scalar(
                select(ReturnRequest).where(ReturnRequest.order_id == order.id, ReturnRequest.sku == sku.upper())
            )
            if existing is not None:
                return _rma_out(existing, "An RMA already exists for this item.")
            customer = await s.get(Customer, customer_id)
            line = next((i for i in order.items if i.sku == sku.strip().upper()), None)
            product = get_product(sku)
            if line is None or product is None or customer is None:
                raise ToolError(f"SKU {sku} is not part of order {order.id}.")
            d = evaluate_return(
                product=product, quantity=line.quantity, unit_price=line.unit_price, tier=customer.tier,  # type: ignore[arg-type]
                order_status=order.status, delivered_at=order.delivered_at,
                opened=line.opened if item_opened is None else item_opened,
                seal_intact=line.seal_intact if seal_intact is None else seal_intact,
                reason=reason, now=utcnow(), auto_approve_limit=settings.agent.refund_auto_approve_limit,
            )  # fmt: skip
            if not d.eligible:
                raise ToolError("Not eligible for return: " + " ".join(d.reasons))
            amount = d.store_credit_amount if resolution == "store_credit" else d.refund_amount
            responses = ctx.input_responses
            if responses is None:
                # Starting a return changes the order and emails a label: the *customer* confirms, with
                # the exact amounts in front of them. Live pass^k evals showed a prompt rule alone
                # ("only on an explicit request") still let an information question create a return.
                fees = f"restocking fee ${d.restocking_fee:.2f}, return shipping ${d.return_shipping_fee:.2f}"
                return InputRequiredResult(
                    result_type="input_required",
                    input_requests={
                        "confirm_return": ElicitRequest(
                            method="elicitation/create",
                            params=ElicitRequestFormParams(
                                message=(
                                    f"Start a return of {line.name} from order {order.id}? You'd receive "
                                    f"${amount:.2f} ({resolution.replace('_', ' ')}; {fees})."
                                ),
                                requested_schema={
                                    "type": "object",
                                    "properties": {"confirm": {"type": "boolean", "title": "Yes, start the return"}},
                                    "required": ["confirm"],
                                },
                            ),
                        )
                    },
                )
            answer = responses.get("confirm_return")
            if not (
                isinstance(answer, ElicitResult)
                and answer.action == "accept"
                and bool((answer.content or {}).get("confirm"))
            ):
                return {"order_id": order.id, "created": False, "message": "The customer chose not to start a return."}
            rma = ReturnRequest(
                id=f"RMA-{secrets.randbelow(90000) + 10000}",
                order_id=order.id,
                customer_id=customer_id,
                sku=line.sku,
                reason=reason,
                resolution=resolution,
                status="label_sent",
                refund_amount=amount,
                restocking_fee=d.restocking_fee,
                return_shipping_fee=d.return_shipping_fee,
                # Damaged-on-arrival items are refunded immediately (KB-004); others on receipt.
                instant_refund_eligible=reason == "damaged_on_arrival",
            )
            rma.label_url = f"https://returns.voltwise.example/labels/{rma.id}.pdf"
            s.add(rma)
            order.status = "return_requested"
            nxt = (
                # Explicit next action (see get_return_status): vague hints like "can be issued" were not acted on.
                f"Refund is due now: call issue_refund(order_id='{order.id}', amount={rma.refund_amount}, "
                "reason='damaged_on_arrival'). Do not tell the customer it is refunded until that call succeeds."
                if rma.instant_refund_eligible
                else "Ship the item with the prepaid label; the refund is processed within 3 business days of receipt."
            )
            return _rma_out(rma, nxt)

    def _rma_out(rma: ReturnRequest, next_steps: str) -> ReturnCreated:
        return ReturnCreated(
            rma_id=rma.id, order_id=rma.order_id, sku=rma.sku, status=rma.status, resolution=rma.resolution,
            refund_amount=rma.refund_amount, instant_refund_eligible=rma.instant_refund_eligible,
            label_url=rma.label_url, next_steps=next_steps,
        )  # fmt: skip

    @mcp.tool(annotations=READ_ONLY, auth=require_scopes("orders:read"), tags={"returns"})
    async def get_return_status(order_id: str, customer_id: str = TokenClaim("sub")) -> list[dict[str, Any]]:
        """Get the status of all returns (RMAs) and refunds for an order, including the next step to take."""
        async with database.session() as s:
            order = await _load_order(s, order_id, customer_id)
            rmas = (await s.scalars(select(ReturnRequest).where(ReturnRequest.order_id == order.id))).all()
            refunds = (await s.scalars(select(Refund).where(Refund.order_id == order.id))).all()

            def next_step(r: ReturnRequest) -> str:
                # Tool results that spell out the next action steer agents far more reliably than
                # a rule buried in the system prompt (a live eval showed the prompt alone was not enough).
                if r.status == "refunded":
                    return "Refund already issued - report the refund details below."
                if r.status == "received" or r.instant_refund_eligible:
                    reason = "damaged_on_arrival" if r.instant_refund_eligible else "return_received"
                    return (
                        f"Refund is due now: call issue_refund(order_id='{order.id}', amount={r.refund_amount}, "
                        f"reason='{reason}'). Amounts over ${settings.agent.refund_auto_approve_limit:.0f} are "
                        "routed to a specialist for approval automatically - still make the call."
                    )
                return (
                    "Waiting for the item: the refund is issued within 3 business days after the warehouse receives it."
                )

            out: list[dict[str, Any]] = [
                {"type": "return", "rma_id": r.id, "sku": r.sku, "status": r.status, "refund_amount": r.refund_amount,
                 "refund_ready": r.status == "received" or r.instant_refund_eligible, "next_step": next_step(r)}
                for r in rmas
            ]  # fmt: skip
            out += [
                {"type": "refund", "refund_id": f.id, "amount": f.amount, "status": f.status,
                 "method": f.method, "issued_at": f.created_at.date().isoformat()}
                for f in refunds
            ]  # fmt: skip
            return out

    async def _price_adjustment(s: Any, order: Order) -> tuple[list[dict[str, Any]], float, float]:
        """(lines, refundable, already paid) - the ONE definition used by ``check_price_adjustment`` and
        ``issue_refund``, so what the agent is told is exactly what a refund accepts. Returned items are
        excluded: an item that comes back is refunded through its return, never adjusted as well."""
        prices = {p.sku: p.price for p in (await s.scalars(select(CatalogPrice))).all()}
        returned = set((await s.scalars(select(ReturnRequest.sku).where(ReturnRequest.order_id == order.id))).all())
        already = float(
            await s.scalar(
                select(func.coalesce(func.sum(Refund.amount), 0.0)).where(
                    Refund.order_id == order.id, Refund.reason == "price_adjustment"
                )
            )
            or 0.0
        )
        lines: list[dict[str, Any]] = [
            {"sku": i.sku, "name": i.name, "paid": i.unit_price, "current_price": prices.get(i.sku, i.unit_price),
             "returned": i.sku in returned,
             "difference": 0.0 if i.sku in returned
             else round(max(i.unit_price - prices.get(i.sku, i.unit_price), 0) * i.quantity, 2)}
            for i in order.items
        ]  # fmt: skip
        refundable = round(max(sum(float(line["difference"]) for line in lines) - already, 0.0), 2)
        return lines, refundable, already

    @mcp.tool(annotations=READ_ONLY, auth=require_scopes("orders:read"), tags={"refunds"})
    async def check_price_adjustment(order_id: str, customer_id: str = TokenClaim("sub")) -> PriceAdjustmentCheck:
        """Check whether the customer is owed a price adjustment because Voltwise lowered the price of an
        item within 14 days of purchase. Returns the exact refundable difference and the next step."""
        async with database.session() as s:
            order = await _load_order(s, order_id, customer_id)
            lines, refundable, already = await _price_adjustment(s, order)
            days = (utcnow() - order.created_at).days
            eligible = days <= 14 and refundable > 0
            if eligible:
                next_step = (
                    f"Price adjustment due: call issue_refund(order_id='{order.id}', amount={refundable}, "
                    "reason='price_adjustment'). Amounts over "
                    f"${settings.agent.refund_auto_approve_limit:.0f} are routed for approval automatically."
                )
            elif days > 14:
                next_step = "Not eligible: price adjustments are only available within 14 days of purchase."
            else:
                next_step = (
                    "Not eligible: the current price is not lower than the price paid (or it was already refunded)."
                )
            return PriceAdjustmentCheck(
                order_id=order.id, eligible=eligible, days_since_purchase=days, lines=lines,
                already_refunded=already, refundable_amount=refundable, next_step=next_step,
            )  # fmt: skip

    @mcp.tool(annotations=DESTRUCTIVE, auth=require_scopes("refunds:write"), tags={"refunds"})
    async def issue_refund(
        order_id: str,
        amount: float,
        reason: Literal["return_received", "damaged_on_arrival", "price_adjustment"],
        method: Literal["original_payment", "store_credit"] = "original_payment",
        approval_code: str | None = None,
        customer_id: str = TokenClaim("sub"),
    ) -> RefundReceipt:
        """Issue a refund. Moves money - irreversible.

        Allowed reasons: `return_received` (the RMA has been received by the warehouse),
        `damaged_on_arrival` (an instant-refund RMA exists) or `price_adjustment` (Voltwise lowered the
        price within 14 days of purchase). The server computes the maximum refundable amount itself.
        Refunds above the auto-approve limit need a human approval; you never provide approval_code yourself."""
        amount = round(float(amount), 2)
        if amount <= 0:
            raise ToolError("Refund amount must be positive.")
        async with database.session() as s:
            order = await _load_order(s, order_id, customer_id)
            already = float(
                await s.scalar(select(func.coalesce(func.sum(Refund.amount), 0.0)).where(Refund.order_id == order.id))
                or 0.0
            )
            rma_id: str | None = None
            if reason in ("return_received", "damaged_on_arrival"):
                rmas = (await s.scalars(select(ReturnRequest).where(ReturnRequest.order_id == order.id))).all()
                ready = [
                    r for r in rmas
                    if r.status == "received" or (reason == "damaged_on_arrival" and r.instant_refund_eligible)
                ]  # fmt: skip
                if not ready:
                    raise ToolError(
                        "No return is ready for refund on this order. Refunds are issued after the warehouse "
                        "receives the item (or immediately for damaged-on-arrival returns)."
                    )
                max_amount = round(sum(r.refund_amount for r in ready) - already, 2)
                rma_id = ready[0].id
            else:  # price_adjustment (KB-007)
                if utcnow() - order.created_at > timedelta(days=14):
                    raise ToolError("Price adjustments are only available within 14 days of purchase.")
                _, max_amount, _ = await _price_adjustment(s, order)
            if max_amount <= 0:
                raise ToolError("Nothing left to refund on this order.")
            if amount > max_amount + 0.005:
                raise ToolError(f"Requested ${amount:.2f} exceeds the refundable amount of ${max_amount:.2f}.")

            approved_by = "auto"
            if amount > settings.agent.refund_auto_approve_limit:
                if not approval_code:
                    raise ToolError(
                        f"Refunds over ${settings.agent.refund_auto_approve_limit:.0f} require approval by a "
                        "support specialist. The request has been routed for review."
                    )
                try:
                    claims = verify_approval_code(
                        settings.mcp, approval_code, customer_id=customer_id, order_id=order.id, amount=amount
                    )
                except ApprovalError as exc:
                    raise ToolError(str(exc)) from exc
                approved_by = f"human:{claims.get('approver', 'unknown')}"

            refund = Refund(
                id=f"RF-{secrets.randbelow(90000) + 10000}",
                order_id=order.id,
                customer_id=customer_id,
                return_id=rma_id,
                amount=amount,
                method=method,
                reason=reason,
                status="processed",
                approved_by=approved_by,
            )
            s.add(refund)
            if method == "store_credit":
                customer = await s.get(Customer, customer_id)
                if customer:
                    customer.store_credit = round(customer.store_credit + amount, 2)
            if rma_id:
                rma = await s.get(ReturnRequest, rma_id)
                if rma:
                    rma.status = "refunded"
                order.status = "refunded"
            return RefundReceipt(
                refund_id=refund.id,
                order_id=order.id,
                amount=amount,
                method=method,
                status=refund.status,
                approved_by=approved_by,
                expected_posting="instantly" if method == "store_credit" else "5-7 business days",
            )

    @mcp.tool(annotations=DESTRUCTIVE, auth=require_scopes("orders:write"), tags={"orders"})
    async def cancel_order(
        order_id: str, ctx: Context, customer_id: str = TokenClaim("sub")
    ) -> dict[str, Any] | InputRequiredResult:
        """Cancel an order that has not shipped yet. The customer is asked to confirm before anything changes."""
        async with database.session() as s:
            order = await _load_order(s, order_id, customer_id)
            if not can_cancel(order.status):
                raise ToolError(
                    f"Order {order.id} is '{order.status}' and can no longer be cancelled. "
                    "It can be returned after delivery instead."
                )
            responses = ctx.input_responses
            if responses is None:
                # Round 1 - ask the *customer* (via elicitation) to confirm. Stateless: nothing is held
                # server-side; the client re-issues the call with the answer attached.
                return InputRequiredResult(
                    result_type="input_required",
                    input_requests={
                        "confirm_cancel": ElicitRequest(
                            method="elicitation/create",
                            params=ElicitRequestFormParams(
                                message=(
                                    f"Cancel order {order.id} ({', '.join(i.name for i in order.items)}, "
                                    f"${order.total:.2f})? This cannot be undone."
                                ),
                                requested_schema={
                                    "type": "object",
                                    "properties": {"confirm": {"type": "boolean", "title": "Yes, cancel my order"}},
                                    "required": ["confirm"],
                                },
                            ),
                        )
                    },
                )
            answer = responses.get("confirm_cancel")
            confirmed = (
                isinstance(answer, ElicitResult)
                and answer.action == "accept"
                and bool((answer.content or {}).get("confirm"))
            )
            if not confirmed:
                return {"order_id": order.id, "cancelled": False, "message": "The customer chose to keep the order."}
            order.status = "cancelled"
            return {
                "order_id": order.id,
                "cancelled": True,
                "message": "Order cancelled. The payment authorization hold is released within 3-5 business days.",
            }

    @mcp.tool(annotations=WRITE_SAFE, auth=require_scopes("orders:write"), tags={"orders"})
    async def update_shipping_address(
        order_id: str, new_address: str, customer_id: str = TokenClaim("sub")
    ) -> dict[str, Any]:
        """Change the shipping address of an order that is still processing (not shipped)."""
        if len(new_address.strip()) < 10:
            raise ToolError("Please provide a complete street address, city, state/region and postal code.")
        async with database.session() as s:
            order = await _load_order(s, order_id, customer_id)
            if order.status != "processing":
                raise ToolError(
                    "The address can only be changed before the order ships. Use the carrier's redirect options."
                )
            order.shipping_address = new_address.strip()
            return {"order_id": order.id, "shipping_address": order.shipping_address, "updated": True}

    # ---- resources & health -----------------------------------------------------------

    @mcp.resource("policy://returns/limits", mime_type="application/json")
    def return_policy_limits() -> dict[str, Any]:
        """Machine-readable return policy constants enforced by this server."""
        return {
            "standard_return_days": 30,
            "plus_return_days": 60,
            "restocking_fee_rate": 0.15,
            "return_shipping_fee": 6.99,
            "store_credit_bonus": 0.10,
            "auto_approve_refund_limit": settings.agent.refund_auto_approve_limit or AUTO_APPROVE_REFUND_LIMIT,
        }

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "commerce"})

    return mcp
