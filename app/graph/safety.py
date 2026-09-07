"""Model-input safety helpers: redact PII and retain only safe commerce fields."""
import re
from typing import Any

_REDACTIONS = (
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[PHONE_REDACTED]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "[EMAIL_REDACTED]"),
    (re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"), "[ID_REDACTED]"),
    (re.compile(r"(?<!\d)(?:\d[ -]?){15,18}\d(?!\d)"), "[PAYMENT_REDACTED]"),
    (re.compile(r"(?:收货地址|地址)\s*[:：]?\s*[^\n，,。；;]{4,80}"), "[ADDRESS_REDACTED]"),
)


def redact_pii(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def order_projection(order: dict[str, Any]) -> dict[str, Any]:
    """Safe subset permitted in Agent state and model context."""
    return {
        "id": order.get("id"),
        "orderSn": order.get("orderSn"),
        "status": order.get("status"),
        "payAmount": order.get("payAmount"),
        "deliverySn": order.get("deliverySn"),
        "orderItemList": [
            {
                "id": item.get("id"),
                "productId": item.get("productId"),
                "productName": item.get("productName"),
                "productQuantity": item.get("productQuantity"),
            }
            for item in order.get("orderItemList", [])
            if isinstance(item, dict)
        ],
    }


def logistics_projection(logistics: dict[str, Any]) -> dict[str, Any]:
    """Safe shipment fields permitted in Agent state and model context."""
    return {
        "orderId": logistics.get("orderId"),
        "orderStatus": logistics.get("orderStatus"),
        "deliveryCompany": logistics.get("deliveryCompany"),
        "deliverySn": logistics.get("deliverySn"),
        "deliveryTime": logistics.get("deliveryTime"),
        "receiveTime": logistics.get("receiveTime"),
    }
