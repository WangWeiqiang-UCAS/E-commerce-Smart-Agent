from uuid import uuid4

import pytest

from app.core.tracing import valid_trace_id
from app.services.commerce_client import CommerceClient


def test_trace_id_is_canonical_uuid_and_rejects_invalid_values():
    trace_id = str(uuid4()).upper()
    assert valid_trace_id(trace_id) == trace_id.lower()
    with pytest.raises(Exception):
        valid_trace_id("not-a-trace")


def test_commerce_client_forwards_trace_id(monkeypatch):
    monkeypatch.setattr("app.services.commerce_client.settings.COMMERCE_API_BASE_URL", "http://commerce")
    trace_id = str(uuid4())

    client = CommerceClient("commerce-token", trace_id)

    assert client._headers == {"Authorization": "Bearer commerce-token", "X-Trace-Id": trace_id}
