"""Subgraph routing, interrupt/resume, and rate-limit tests."""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langgraph.types import Command

from app.core.rate_limit import check_rate_limit
from app.graph.workflow import compile_memory_graph, route_intent


def _fake_llm(intent, answer="Answer from fake LLM."):
    from app.graph.nodes import IntentDecision

    class FakeClassifier:
        async def ainvoke(self, messages):
            return IntentDecision(intent=intent)

    class FakeLLM:
        def with_structured_output(self, schema):
            return FakeClassifier()

        async def ainvoke(self, messages):
            return SimpleNamespace(content=answer)

    return FakeLLM()


def _initial_state(**overrides):
    state = {
        "question": "order status SN9",
        "user_id": 1,
        "thread_id": "user_1_thread_1",
        "history": [],
        "context": [],
        "order_data": None,
        "products": [],
        "after_sale_data": None,
        "handoff": None,
        "answer": "",
    }
    state.update(overrides)
    return state


def test_router_maps_all_intents():
    assert route_intent({"intent": "GUIDE"}) == "guide_subgraph"
    assert route_intent({"intent": "ORDER"}) == "order_subgraph"
    assert route_intent({"intent": "POLICY"}) == "policy_subgraph"
    assert route_intent({"intent": "REFUND"}) == "after_sale_subgraph"
    assert route_intent({"intent": "OTHER"}) == "fallback_subgraph"
    assert route_intent({}) == "fallback_subgraph"


@pytest.mark.asyncio
async def test_refund_flow_interrupts_then_resumes_with_handoff(monkeypatch):
    from app.graph import nodes

    class FakeCommerce:
        def __init__(self, *args, **kwargs):
            pass

        async def get_orders(self):
            return [{
                "id": 9, "orderSn": "SN9", "status": 2, "payAmount": "19.99", "deliverySn": "TRACK9",
                "orderItemList": [{"id": 1, "productId": 2, "productName": "Shoe", "productQuantity": 2}],
            }]

        async def get_after_sale_status(self, after_sale_id):
            raise AssertionError("status lookup should not run for refund intent")

    monkeypatch.setattr(nodes, "llm", _fake_llm("REFUND", "The refund preview handoff is ready."))
    monkeypatch.setattr(nodes, "CommerceClient", FakeCommerce)

    graph = compile_memory_graph()
    config = {"configurable": {"thread_id": "user_1_thread_1"}}
    state = _initial_state(question="我要退货，订单号 SN9，买 2 件，因为尺码不合适", intent="REFUND")

    result = await graph.ainvoke(state, config)
    interrupts = result.get("__interrupt__")
    assert interrupts, "refund intent must pause for explicit confirmation"
    payload = interrupts[0].value
    assert payload["kind"] == "refund_preview_confirmation"
    assert payload["draft"]["quantity"] == 2
    assert payload["draft"]["productName"] == "Shoe"

    resumed = await graph.ainvoke(
        Command(resume={"decision": "confirmed", "confirmed_at": "2026-08-08T12:00:00Z"}),
        config,
    )
    assert resumed.get("handoff") == {
        "orderId": 9, "orderSn": "SN9", "productId": 2, "productName": "Shoe",
        "quantity": 2, "reason": "尺码不合适",
    }
    assert resumed.get("refund_flow_active") is True


@pytest.mark.asyncio
async def test_refund_flow_cancel_does_not_create_handoff(monkeypatch):
    from app.graph import nodes

    class FakeCommerce:
        def __init__(self, *args, **kwargs):
            pass

        async def get_orders(self):
            return [{
                "id": 9, "orderSn": "SN9", "status": 2, "payAmount": "19.99", "deliverySn": "TRACK9",
                "orderItemList": [{"id": 1, "productId": 2, "productName": "Shoe", "productQuantity": 1}],
            }]

        async def get_after_sale_status(self, after_sale_id):
            raise AssertionError("status lookup should not run for refund intent")

    monkeypatch.setattr(nodes, "llm", _fake_llm("REFUND", "cancelled"))
    monkeypatch.setattr(nodes, "CommerceClient", FakeCommerce)

    graph = compile_memory_graph()
    config = {"configurable": {"thread_id": "user_1_thread_2"}}
    state = _initial_state(question="取消退款申请，订单号 SN9", intent="REFUND")

    result = await graph.ainvoke(state, config)
    assert result.get("__interrupt__")

    resumed = await graph.ainvoke(Command(resume={"decision": "cancelled"}), config)
    assert resumed.get("handoff") is None
    assert resumed.get("refund_flow_active") is False
    assert "cancelled" in resumed.get("answer", "").lower()


@pytest.mark.asyncio
async def test_guide_subgraph_uses_commerce_search(monkeypatch):
    from app.graph import nodes

    class FakeCommerce:
        def __init__(self, *args, **kwargs):
            pass

        async def search_products(self, query):
            return [{"id": 3, "name": "Running Shoe", "price": 299.0, "promotionPrice": 199.0,
                     "stock": 8, "brandName": "Nike", "categoryName": "Shoes",
                     "description": "secret marketing copy", "pic": "https://example.invalid/pic.png"}]

    monkeypatch.setattr(nodes, "llm", _fake_llm("GUIDE", "Here are matching products."))
    monkeypatch.setattr(nodes, "CommerceClient", FakeCommerce)

    graph = compile_memory_graph()
    result = await graph.ainvoke(_initial_state(question="推荐几双跑鞋", intent="GUIDE"), {"configurable": {"thread_id": "t_guide"}})
    assert result["products"][0]["name"] == "Running Shoe"
    assert "description" not in result["products"][0]
    assert result["answer"] == "Here are matching products."


@pytest.mark.asyncio
async def test_interrupt_is_detectable_via_checkpoint_state(monkeypatch):
    from app.graph import nodes

    class FakeCommerce:
        def __init__(self, *args, **kwargs):
            pass

        async def get_orders(self):
            return [{
                "id": 9, "orderSn": "SN9", "status": 2, "payAmount": "19.99",
                "orderItemList": [{"id": 1, "productId": 2, "productName": "Shoe", "productQuantity": 1}],
            }]

        async def get_after_sale_status(self, after_sale_id):
            raise AssertionError("status lookup should not run for refund intent")

    monkeypatch.setattr(nodes, "llm", _fake_llm("REFUND"))
    monkeypatch.setattr(nodes, "CommerceClient", FakeCommerce)

    graph = compile_memory_graph()
    config = {"configurable": {"thread_id": "user_1_thread_3"}}
    state = _initial_state(question="我要退货，订单号 SN9", intent="REFUND")

    async for _ in graph.astream_events(state, config, version="v2"):
        pass

    snapshot = await graph.aget_state(config)
    assert snapshot.next, "graph must be paused at the interrupt"
    payloads = [pending.value for task in snapshot.tasks for pending in task.interrupts]
    assert payloads and payloads[0]["kind"] == "refund_preview_confirmation"


def test_rate_limiter_blocks_excess_requests(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 3)
    for _ in range(3):
        check_rate_limit("unit:test")
    with pytest.raises(HTTPException) as error:
        check_rate_limit("unit:test")
    assert error.value.status_code == 429


def test_rate_limiter_can_be_disabled(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 1)
    for _ in range(5):
        check_rate_limit("unit:disabled")
