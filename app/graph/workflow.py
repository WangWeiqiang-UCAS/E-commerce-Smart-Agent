import redis.asyncio as redis
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import AsyncRedisSaver
from app.graph.state import AgentState
from app.graph.nodes import retrieve, generate, intent_router, query_order
from app.core.config import settings

# 1. 定义路由逻辑 (保持不变)
def route_intent(state: AgentState):
    intent = state.get("intent")
    if intent == "ORDER":
        return "query_order"
    elif intent == "POLICY":
        return "retrieve"
    return "generate"

# 2. 构建图 (只定义结构，不编译)
workflow = StateGraph(AgentState)

workflow.add_node("intent_router", intent_router)
workflow.add_node("retrieve", retrieve)
workflow.add_node("query_order", query_order)
workflow.add_node("generate", generate)

workflow.add_edge(START, "intent_router")

workflow.add_conditional_edges(
    "intent_router",
    route_intent,
    {
        "query_order": "query_order",
        "retrieve": "retrieve",
        "generate": "generate"
    }
)

# 之前这里的错误：workflow.add_edge("retrieve", "retrieve")
workflow.add_edge("query_order", "generate")
workflow.add_edge("retrieve", "generate") # 修正
workflow.add_edge("generate", END)


async def compile_app_graph():
    # 不再需要 global app_graph
    checkpointer = AsyncRedisSaver(redis_url=settings.REDIS_URL)
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    print("✅ LangGraph app_graph compiled with AsyncRedisSaver.")
    return compiled_graph # 返回编译后的图

