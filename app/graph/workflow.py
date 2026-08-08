"""Deterministic agent workflow with commerce-owned state changes.

One router plus focused, deterministic subgraphs (guided selling, orders,
policy, after-sales). Refund write intent pauses at an explicit user
confirmation via LangGraph ``interrupt`` and resumes with a handoff payload;
no commerce write is available to the model.
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.graph.nodes import (
    generate,
    guide,
    handle_after_sale,
    intent_router,
    query_order,
    retrieve,
)
from app.graph.state import AgentState

app_graph = None


def _subgraph(node_name: str, node, *, with_generate: bool = True):
    """Build a focused subgraph: domain node -> generate -> END."""
    graph = StateGraph(AgentState)
    graph.add_node(node_name, node)
    if with_generate:
        graph.add_node("generate", generate)
        graph.add_edge(START, node_name)
        graph.add_edge(node_name, "generate")
        graph.add_edge("generate", END)
    else:
        graph.add_edge(START, node_name)
        graph.add_edge(node_name, END)
    return graph.compile()


guide_subgraph = _subgraph("guide", guide)
order_subgraph = _subgraph("query_order", query_order)
policy_subgraph = _subgraph("retrieve", retrieve)
after_sale_subgraph = _subgraph("handle_after_sale", handle_after_sale)
fallback_subgraph = _subgraph("generate", generate, with_generate=False)


def route_intent(state: AgentState) -> str:
    intent = state.get("intent")
    return {
        "GUIDE": "guide_subgraph",
        "ORDER": "order_subgraph",
        "POLICY": "policy_subgraph",
        "REFUND": "after_sale_subgraph",
    }.get(intent, "fallback_subgraph")


workflow = StateGraph(AgentState)
workflow.add_node("intent_router", intent_router)
workflow.add_node("guide_subgraph", guide_subgraph)
workflow.add_node("order_subgraph", order_subgraph)
workflow.add_node("policy_subgraph", policy_subgraph)
workflow.add_node("after_sale_subgraph", after_sale_subgraph)
workflow.add_node("fallback_subgraph", fallback_subgraph)

workflow.add_edge(START, "intent_router")
workflow.add_conditional_edges(
    "intent_router",
    route_intent,
    {
        "guide_subgraph": "guide_subgraph",
        "order_subgraph": "order_subgraph",
        "policy_subgraph": "policy_subgraph",
        "after_sale_subgraph": "after_sale_subgraph",
        "fallback_subgraph": "fallback_subgraph",
    },
)
for subgraph in ("guide_subgraph", "order_subgraph", "policy_subgraph", "after_sale_subgraph", "fallback_subgraph"):
    workflow.add_edge(subgraph, END)


def compile_app_graph(saver=None):
    """Compile the workflow with a checkpoint saver.

    Production uses Redis (AsyncRedisSaver); tests inject an in-memory saver so
    interrupt/resume can be exercised without infrastructure.
    """
    checkpointer = saver or AsyncRedisSaver(redis_url=settings.REDIS_URL)
    return workflow.compile(checkpointer=checkpointer)


def compile_memory_graph():
    """Compile with a synchronous in-memory saver (tests, local demos)."""
    return compile_app_graph(MemorySaver())
