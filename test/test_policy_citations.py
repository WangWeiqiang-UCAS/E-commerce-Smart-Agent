import pytest

from app.graph import nodes
from app.graph.nodes import _policy_citations, _tool_citations


def test_policy_citations_preserve_document_version_and_chunk_identity():
    citations = _policy_citations([
        "Policy text\n[citation source=returns.md; version=v3; chunk_id=12; document_id=returns-policy]",
        "Other text\n[citation source=returns.md; version=v3; chunk_id=12; document_id=returns-policy]",
    ])

    assert citations == ["[citation source=returns.md; version=v3; chunk_id=12; document_id=returns-policy]"]


def test_policy_citations_never_extracts_unstructured_model_like_text():
    assert _policy_citations(["source=made-up; version=v0; chunk_id=99"]) == []


def test_tool_citations_use_only_deterministic_commerce_identifiers():
    citations = _tool_citations({
        "order_data": {"orderSn": "SN20260820001", "memberId": 99},
        "order_detail_loaded": True,
        "products": [{"name": "Keyboard"}, {"name": "Mouse"}],
        "product_detail": {"product": {"id": 8}},
        "after_sale_data": {"afterSaleId": 42, "reason": "private"},
    })

    assert citations == [
        "[tool name=get_user_orders; order_reference=SN20260820001]",
        "[tool name=get_order_detail; order_id=unknown]",
        "[tool name=search_products; result_count=2]",
        "[tool name=get_product_detail; product_id=8]",
        "[tool name=get_after_sale_status; after_sale_id=42]",
    ]


@pytest.mark.asyncio
async def test_generate_does_not_invoke_the_model_without_verified_evidence(monkeypatch):
    class UnexpectedModelCall:
        async def ainvoke(self, _messages):
            raise AssertionError("empty-evidence fallback must not call the model")

    monkeypatch.setattr(nodes, "llm", UnexpectedModelCall())

    result = await nodes.generate({"question": "What is the best current offer?", "context": []})

    assert result == {"answer": nodes.NO_VERIFIED_EVIDENCE_ANSWER}
