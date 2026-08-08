import asyncio

from app.services.commerce_client import CommerceAPIError, CommerceClient


def test_logistics_and_after_sale_use_allow_listed_paths():
    client = object.__new__(CommerceClient)
    requests = []

    async def request(method, path, **kwargs):
        requests.append((method, path))
        return {"orderId": 7} if "logistics" in path else {"afterSaleId": 11}

    client._request = request

    assert asyncio.run(client.get_logistics(7)) == {"orderId": 7}
    assert asyncio.run(client.get_after_sale_status(11)) == {"afterSaleId": 11}
    assert requests == [
        ("GET", "/agent/commerce/orders/7/logistics"),
        ("GET", "/agent/commerce/after-sales/11"),
    ]


def test_read_tools_reject_invalid_commerce_payloads():
    client = object.__new__(CommerceClient)

    async def request(method, path, **kwargs):
        return []

    client._request = request

    try:
        asyncio.run(client.get_after_sale_status(11))
    except CommerceAPIError as exc:
        assert "after-sales" in str(exc)
    else:
        raise AssertionError("Expected invalid response to be rejected")


def test_product_tools_use_bounded_allow_listed_paths():
    client = object.__new__(CommerceClient)
    requests = []

    async def request(method, path, **kwargs):
        requests.append((method, path, kwargs.get("params")))
        return [{"id": 9}] if path.endswith("/search") else {"product": {"id": 9}}

    client._request = request

    assert asyncio.run(client.search_products("shoe", page_num=0, page_size=999, sort=99)) == [{"id": 9}]
    assert asyncio.run(client.get_product_detail(9)) == {"product": {"id": 9}}
    assert requests == [
        ("GET", "/agent/commerce/products/search", {"query": "shoe", "brandId": None, "categoryId": None, "pageNum": 1, "pageSize": 20, "sort": 4}),
        ("GET", "/agent/commerce/products/9", None),
    ]
