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


def test_identity_is_read_from_the_allow_listed_commerce_boundary():
    client = object.__new__(CommerceClient)
    calls = []

    async def request(method, path, **kwargs):
        calls.append((method, path))
        return {"memberId": 42, "username": "member"}

    client._request = request

    assert asyncio.run(client.get_identity()) == {"memberId": 42, "username": "member"}
    assert calls == [("GET", "/agent/commerce/identity")]


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


def test_client_rejects_non_allow_listed_or_write_paths_before_network_io():
    client = object.__new__(CommerceClient)

    try:
        asyncio.run(client._request("POST", "/agent/commerce/actions/refund/confirm"))
    except CommerceAPIError as exc:
        assert "allow-listed" in str(exc)
    else:
        raise AssertionError("Write paths must not be callable by the Agent client")

    try:
        asyncio.run(client._request("GET", "/actuator/env"))
    except CommerceAPIError as exc:
        assert "allow-listed" in str(exc)
    else:
        raise AssertionError("Unexpected paths must be rejected before network IO")


def test_client_retries_transient_gateway_failure(monkeypatch):
    monkeypatch.setattr("app.services.commerce_client.settings.COMMERCE_API_BASE_URL", "http://commerce")
    monkeypatch.setattr("app.services.commerce_client.settings.COMMERCE_API_READ_RETRY_ATTEMPTS", 2)
    attempts = []

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code

        def json(self):
            return {"code": 200, "data": {"list": []}}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def request(self, *_args, **_kwargs):
            attempts.append(1)
            return Response(503 if len(attempts) == 1 else 200)

    monkeypatch.setattr("app.services.commerce_client.httpx.AsyncClient", FakeClient)

    assert asyncio.run(CommerceClient("token").get_orders()) == []
    assert len(attempts) == 2
