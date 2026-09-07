"""Allow-listed, read-only client for the Java commerce anti-corruption API."""
import re
from time import perf_counter
from typing import Any, Optional

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from app.core.config import settings
from app.core.metrics import record_commerce_call
from app.core.telemetry import commerce_tracer


class CommerceAPIError(RuntimeError):
    """A controlled commerce API failure safe to present as an unavailable tool."""


class _RetryableCommerceError(RuntimeError):
    """Internal marker for transient read failures only."""


class CommerceClient:
    _READ_PATHS = (
        re.compile(r"^/agent/commerce/orders$"),
        re.compile(r"^/agent/commerce/identity$"),
        re.compile(r"^/agent/commerce/orders/\d+$"),
        re.compile(r"^/agent/commerce/orders/\d+/logistics$"),
        re.compile(r"^/agent/commerce/after-sales/\d+$"),
        re.compile(r"^/agent/commerce/products/search$"),
        re.compile(r"^/agent/commerce/products/\d+$"),
    )
    def __init__(self, access_token: str, trace_id: Optional[str] = None):
        if not settings.COMMERCE_API_BASE_URL:
            raise CommerceAPIError("Commerce integration is not configured")
        if not access_token:
            raise CommerceAPIError("A commerce access token is required")
        self._base_url = settings.COMMERCE_API_BASE_URL.rstrip("/")
        self._headers = {"Authorization": f"Bearer {access_token}"}
        if trace_id:
            self._headers["X-Trace-Id"] = trace_id

    async def get_orders(self, status: int = -1, page_num: int = 1, page_size: int = 50) -> list[dict[str, Any]]:
        data = await self._request("GET", "/agent/commerce/orders", params={
            "status": status, "pageNum": page_num, "pageSize": min(max(page_size, 1), 50),
        })
        return data.get("list", []) if isinstance(data, dict) else []

    async def get_identity(self) -> dict[str, Any]:
        data = await self._request("GET", "/agent/commerce/identity")
        if not isinstance(data, dict) or not isinstance(data.get("memberId"), int):
            raise CommerceAPIError("Commerce returned an invalid member identity")
        return data

    async def search_products(self, query: str, *, brand_id: Optional[int] = None,
                              category_id: Optional[int] = None, page_num: int = 1,
                              page_size: int = 10, sort: int = 0) -> list[dict[str, Any]]:
        data = await self._request("GET", "/agent/commerce/products/search", params={
            "query": query, "brandId": brand_id, "categoryId": category_id,
            "pageNum": max(page_num, 1), "pageSize": min(max(page_size, 1), 20),
            "sort": min(max(sort, 0), 4),
        })
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise CommerceAPIError("Commerce returned an invalid product search response")
        return data

    async def get_product_detail(self, product_id: int) -> dict[str, Any]:
        data = await self._request("GET", f"/agent/commerce/products/{product_id}")
        if not isinstance(data, dict):
            raise CommerceAPIError("Commerce returned an invalid product detail response")
        return data

    async def get_order(self, order_id: int) -> dict[str, Any]:
        data = await self._request("GET", f"/agent/commerce/orders/{order_id}")
        if not isinstance(data, dict):
            raise CommerceAPIError("Commerce returned an invalid order response")
        return data

    async def get_logistics(self, order_id: int) -> dict[str, Any]:
        data = await self._request("GET", f"/agent/commerce/orders/{order_id}/logistics")
        if not isinstance(data, dict):
            raise CommerceAPIError("Commerce returned an invalid logistics response")
        return data

    async def get_after_sale_status(self, after_sale_id: int) -> dict[str, Any]:
        data = await self._request("GET", f"/agent/commerce/after-sales/{after_sale_id}")
        if not isinstance(data, dict):
            raise CommerceAPIError("Commerce returned an invalid after-sales response")
        return data

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        if method != "GET" or not any(pattern.fullmatch(path) for pattern in self._READ_PATHS):
            raise CommerceAPIError("Commerce tool path is not allow-listed")
        started = perf_counter()
        outcome = "error"
        tool_name = self._tool_name(path)
        with commerce_tracer().start_as_current_span("commerce.tool.read") as span:
            span.set_attribute("commerce.tool.name", tool_name)
            span.set_attribute("http.request.method", method)
            try:
                try:
                    async for attempt in AsyncRetrying(
                        retry=retry_if_exception_type(_RetryableCommerceError),
                        stop=stop_after_attempt(max(1, settings.COMMERCE_API_READ_RETRY_ATTEMPTS)),
                        wait=wait_random_exponential(multiplier=0.1, max=0.5),
                        reraise=True,
                    ):
                        with attempt:
                            async with httpx.AsyncClient(timeout=settings.COMMERCE_API_TIMEOUT_SECONDS) as client:
                                response = await client.request(method, f"{self._base_url}{path}", headers=self._headers, **kwargs)
                            if response.status_code in {502, 503, 504}:
                                raise _RetryableCommerceError()
                except (httpx.TimeoutException, httpx.TransportError, _RetryableCommerceError) as exc:
                    raise CommerceAPIError("Commerce service is temporarily unavailable") from exc
                span.set_attribute("http.response.status_code", response.status_code)
                if response.status_code in {401, 403, 404}:
                    raise CommerceAPIError("Commerce record was not found or is not available to this user")
                if response.status_code >= 400:
                    raise CommerceAPIError("Commerce service rejected the request")
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise CommerceAPIError("Commerce service returned an invalid response") from exc
                if not isinstance(payload, dict) or payload.get("code") not in {200, 0}:
                    raise CommerceAPIError(payload.get("message", "Commerce service request failed") if isinstance(payload, dict) else "Commerce service request failed")
                outcome = "success"
                return payload.get("data")
            finally:
                span.set_attribute("commerce.outcome", outcome)
                record_commerce_call(tool_name, outcome, perf_counter() - started)

    @staticmethod
    def _tool_name(path: str) -> str:
        if path.endswith("/logistics"):
            return "get_logistics"
        if "/after-sales/" in path:
            return "get_after_sale_status"
        if path.endswith("/products/search"):
            return "search_products"
        if "/products/" in path:
            return "get_product_detail"
        if path == "/agent/commerce/orders":
            return "get_orders"
        if path == "/agent/commerce/identity":
            return "get_identity"
        return "get_order"
