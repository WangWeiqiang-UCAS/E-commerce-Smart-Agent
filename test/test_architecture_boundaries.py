from pathlib import Path

from sqlmodel import SQLModel

import app.models  # noqa: F401 - registers current runtime models


ROOT = Path(__file__).resolve().parents[1]


def test_agent_runtime_metadata_has_no_commerce_system_of_record_tables():
    forbidden = {"users", "orders", "refund_applications", "order_items"}
    assert not forbidden.intersection(SQLModel.metadata.tables)
    assert "support_operators" in SQLModel.metadata.tables
    assert "inbox_events" in SQLModel.metadata.tables


def test_customer_bff_does_not_keep_a_second_agent_customer_token():
    source = (ROOT / "apps/agent-bff-ts/src/lib/server-auth.ts").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
    assert "AGENT_SESSION_COOKIE" not in source
    assert "AGENT_SESSION_COOKIE" not in compose
    assert "commerce_access_token" in source


def test_python_customer_registration_endpoint_is_not_present():
    source = (ROOT / "app/api/v1/auth.py").read_text(encoding="utf-8")
    assert '@router.post("/register"' not in source
    assert "SupportOperator" in source


def test_customer_widget_fetches_product_details_only_through_the_bff():
    source = (ROOT / "apps/customer-widget-ts/src/widget.tsx").read_text(encoding="utf-8")

    assert 'fetch(`/api/products/${encodeURIComponent(String(productId))}`' in source
    assert "COMMERCE_API_URL" not in source
    assert "/agent/commerce/" not in source


def test_desk_keeps_agent_suggestions_as_human_submitted_drafts():
    source = (ROOT / "apps/agent-desk-ts/src/app/page.tsx").read_text(encoding="utf-8")

    assert 'message.sender_type === "agent"' in source
    assert "Use as editable reply" in source
    assert "Review and submit this reply yourself." in source
