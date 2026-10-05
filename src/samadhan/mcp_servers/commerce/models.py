"""Commerce data model: customers, orders, shipments, returns (RMAs) and refunds."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from samadhan.mcp_servers.db import utcnow


class CommerceBase(DeclarativeBase):
    """Separate metadata per service: each MCP server owns its own database."""


class Customer(CommerceBase):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(200), unique=True)
    tier: Mapped[str] = mapped_column(String(16), default="standard")  # standard | plus
    country: Mapped[str] = mapped_column(String(2), default="US")
    member_since: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    voltpoints: Mapped[int] = mapped_column(Integer, default=0)
    store_credit: Mapped[float] = mapped_column(Float, default=0.0)
    default_address: Mapped[str] = mapped_column(String(300))

    orders: Mapped[list[Order]] = relationship(back_populates="customer")


class Order(CommerceBase):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    shipped_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    shipping_method: Mapped[str] = mapped_column(String(16), default="standard")
    shipping_address: Mapped[str] = mapped_column(String(300))
    subtotal: Mapped[float] = mapped_column(Float)
    shipping_cost: Mapped[float] = mapped_column(Float, default=0.0)
    tax: Mapped[float] = mapped_column(Float, default=0.0)
    total: Mapped[float] = mapped_column(Float)
    payment_method: Mapped[str] = mapped_column(String(40))

    customer: Mapped[Customer] = relationship(back_populates="orders")
    items: Mapped[list[OrderItem]] = relationship(back_populates="order", lazy="selectin")
    shipment: Mapped[Shipment | None] = relationship(back_populates="order", lazy="selectin", uselist=False)


class OrderItem(CommerceBase):
    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), index=True)
    sku: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(120))
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    unit_price: Mapped[float] = mapped_column(Float)
    # Condition as declared by the customer (used as the default for eligibility checks).
    opened: Mapped[bool] = mapped_column(Boolean, default=False)
    seal_intact: Mapped[bool] = mapped_column(Boolean, default=True)

    order: Mapped[Order] = relationship(back_populates="items")


class Shipment(CommerceBase):
    __tablename__ = "shipments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), unique=True)
    carrier: Mapped[str] = mapped_column(String(16))
    tracking_number: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(24))
    estimated_delivery: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    events: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)

    order: Mapped[Order] = relationship(back_populates="shipment")


class ReturnRequest(CommerceBase):
    __tablename__ = "returns"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # RMA-xxxxx
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), index=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    sku: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(String(32))
    resolution: Mapped[str] = mapped_column(String(16))  # refund | store_credit | exchange
    status: Mapped[str] = mapped_column(String(24))  # label_sent | in_transit | received | refunded | closed
    refund_amount: Mapped[float] = mapped_column(Float, default=0.0)
    restocking_fee: Mapped[float] = mapped_column(Float, default=0.0)
    return_shipping_fee: Mapped[float] = mapped_column(Float, default=0.0)
    instant_refund_eligible: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    label_url: Mapped[str] = mapped_column(String(200), default="")


class Refund(CommerceBase):
    __tablename__ = "refunds"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # RF-xxxxx
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), index=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    return_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    amount: Mapped[float] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(24))  # original_payment | store_credit
    reason: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="processed")
    approved_by: Mapped[str] = mapped_column(String(80), default="auto")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class CatalogPrice(CommerceBase):
    """Current selling price (differs from the catalog list price during promotions)."""

    __tablename__ = "catalog_prices"

    sku: Mapped[str] = mapped_column(String(32), primary_key=True)
    price: Mapped[float] = mapped_column(Float)
    stock: Mapped[int] = mapped_column(Integer, default=100)
