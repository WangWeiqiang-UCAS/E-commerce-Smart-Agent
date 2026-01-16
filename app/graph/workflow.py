import redis.asyncio as redis
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import AsyncRedisSaver
from app.graph.state import AgentState
from app.graph.nodes import retrieve, generate, intent_router, query_order
from app.core.config import settings

# 🔑 关键：声明全局变量，供 main.py 和 chat.py 使用
app_graph = None

# 1. 定义路由逻辑
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

workflow.add_edge("query_order", "generate")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)


async def compile_app_graph():
    """
    编译 LangGraph，初始化 Redis checkpointer
    """
    try:
        # 1. 测试 Redis 连接
        print("🔧 测试 Redis 连接...")
        redis_client = redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        print("✅ Redis 连接成功")
        await redis_client.close()
        
        # 2. 创建 checkpointer（传递 URL 字符串）
        checkpointer = AsyncRedisSaver(settings.REDIS_URL)
        
        # 3. 初始化 Redis 索引（关键步骤！）
        print("🔧 初始化 Redis checkpoint 索引...")
        await checkpointer.setup()
        print("✅ Redis checkpoint 索引初始化完成")
        
        # 4. 编译图
        compiled_graph = workflow.compile(checkpointer=checkpointer)
        print("✅ LangGraph 编译完成")
        
        return compiled_graph
        
    except Exception as e:
        print(f"❌ 编译失败: {e}")
        import traceback
        traceback.print_exc()
        raise