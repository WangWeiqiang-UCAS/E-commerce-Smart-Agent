"""Allow-listed client for the Java commerce anti-corruption API."""
from typing import Any, Optional

import httpx

from app.core.config import settings


class CommerceAPIError(RuntimeError):
    """A controlled commerce API failure safe to present as an unavailable tool."""


class CommerceClient:
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
        try:
            async with httpx.AsyncClient(timeout=settings.COMMERCE_API_TIMEOUT_SECONDS) as client:
                response = await client.request(method, f"{self._base_url}{path}", headers=self._headers, **kwargs)
        except httpx.HTTPError as exc:
            raise CommerceAPIError("Commerce service is temporarily unavailable") from exc
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
        return payload.get("data")
