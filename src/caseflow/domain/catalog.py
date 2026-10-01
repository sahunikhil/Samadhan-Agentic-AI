"""Voltwise product catalog (mirrors the knowledge base articles)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Category = Literal["laptop", "tablet", "phone", "headphones", "earbuds", "speaker", "accessory", "gift_card"]


@dataclass(frozen=True, slots=True)
class Product:
    sku: str
    name: str
    category: Category
    price: float
    warranty_months: int
    returnable: bool = True
    hygiene_sensitive: bool = False  # in-ear products: returnable only with intact seal

    @property
    def restocking_fee_applies(self) -> bool:
        return self.category in ("laptop", "tablet")


PRODUCTS: dict[str, Product] = {
    p.sku: p
    for p in [
        Product("VB-AIR-14", "VoltBook Air 14", "laptop", 1099.00, 24),
        Product("VB-PRO-16", "VoltBook Pro 16", "laptop", 1899.00, 24),
        Product("NOVA-8-128", "Nova 8 (128GB)", "phone", 699.00, 12),
        Product("NOVA-8-256", "Nova 8 (256GB)", "phone", 799.00, 12),
        Product("NOVA-TAB-11", "Nova Tab 11", "tablet", 449.00, 12),
        Product("PULSE-ANC", "Pulse ANC Headphones", "headphones", 249.00, 12),
        Product("PULSE-BUDS", "Pulse Buds Pro", "earbuds", 179.00, 12, hygiene_sensitive=True),
        Product("PULSE-CUSH", "Pulse ANC Replacement Ear Cushions", "accessory", 29.00, 12),
        Product("AURA-SPK", "Aura Smart Speaker", "speaker", 129.00, 12),
        Product("VC-65W", "VoltCharge 65W GaN Charger", "accessory", 49.00, 12),
        Product("VC-140W", "VoltCharge 140W GaN Charger", "accessory", 89.00, 12),
        Product("PC-20K", "PowerCell 20K Power Bank", "accessory", 59.00, 12),
        Product("VOLT-DOCK", "Volt Dock USB4", "accessory", 199.00, 12),
        Product("GIFT-50", "Voltwise Gift Card ($50)", "gift_card", 50.00, 0, returnable=False),
    ]
}


def get_product(sku: str) -> Product | None:
    return PRODUCTS.get(sku.upper().strip())
