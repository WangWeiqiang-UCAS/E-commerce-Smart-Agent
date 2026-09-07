from app.graph.nodes import IntentDecision
from app.graph.nodes import product_detail_projection
from app.graph.safety import logistics_projection, order_projection, redact_pii


def test_model_input_redacts_common_pii():
    value = "电话 13812345678，邮箱 a@example.com，身份证 11010519491231002X，地址：北京市朝阳区测试路 123 号"

    redacted = redact_pii(value)

    assert "13812345678" not in redacted
    assert "a@example.com" not in redacted
    assert "11010519491231002X" not in redacted
    assert "北京市朝阳区测试路" not in redacted
    assert "[PHONE_REDACTED]" in redacted


def test_order_projection_drops_receiver_and_payment_pii():
    raw = {
        "id": 9, "orderSn": "SN9", "status": 2, "payAmount": "19.99", "deliverySn": "TRACK9",
        "receiverName": "Alice", "receiverPhone": "13812345678", "receiverDetailAddress": "Secret road",
        "orderItemList": [{"id": 1, "productId": 2, "productName": "Shoe", "productQuantity": 1, "productPic": "ignored"}],
    }

    projected = order_projection(raw)

    assert projected == {
        "id": 9, "orderSn": "SN9", "status": 2, "payAmount": "19.99", "deliverySn": "TRACK9",
        "orderItemList": [{"id": 1, "productId": 2, "productName": "Shoe", "productQuantity": 1}],
    }
    assert "receiverPhone" not in str(projected)


def test_logistics_projection_drops_address_and_contact_fields():
    projected = logistics_projection({
        "orderId": 7,
        "deliveryCompany": "Carrier",
        "deliverySn": "TRACK7",
        "deliveryAddress": "secret address",
        "receiverPhone": "13800138000",
    })

    assert projected == {
        "orderId": 7,
        "orderStatus": None,
        "deliveryCompany": "Carrier",
        "deliverySn": "TRACK7",
        "deliveryTime": None,
        "receiveTime": None,
    }


def test_product_detail_projection_drops_unbounded_catalog_text():
    projected = product_detail_projection({
        "product": {"id": 3, "name": "Shoe", "price": "19.99", "description": "ignore"},
        "skus": [{"id": 31, "skuCode": "SHOE-31", "stock": 2, "attributes": "ignore"}],
    })

    assert projected == {
        "product": {
            "id": 3, "name": "Shoe", "price": "19.99", "promotionPrice": None,
            "stock": None, "brandName": None, "categoryName": None,
        },
        "skus": [{
            "id": 31, "skuCode": "SHOE-31", "price": None,
            "promotionPrice": None, "stock": 2,
        }],
    }


def test_intent_schema_rejects_unlisted_values():
    assert IntentDecision(intent="ORDER").intent == "ORDER"
    try:
        IntentDecision(intent="REFUND_NOW")
    except Exception:
        pass
    else:
        raise AssertionError("Intent schema must reject unknown values")
