# scripts/test_v2_logic.py
import asyncio
import json
from app.core.config import settings
from app.core.database import init_db
from app.core.security import create_access_token
# 导入 compile_app_graph 函数，但不再导入 app_graph 变量
from app.graph.workflow import compile_app_graph
from langchain_core.runnables import RunnableConfig # 导入 RunnableConfig

# 在测试脚本中，我们需要一个局部变量来保存编译后的 app_graph
_test_app_graph = None # 使用前缀，避免与模块中的 app_graph 混淆

async def simulate_agent_call(query: str, user_id: int, thread_id: str):
    """
    模拟调用 Agent 的逻辑
    """
    global _test_app_graph # 声明会修改全局变量
    
    print(f"\n--- [User ID: {user_id} | Thread: {thread_id}] ---")
    print(f"❓ Question: {query}")
    
    initial_state = {
        "question": query,
        "user_id": user_id,
        "history": [],
        "context": [],
        "order_data": None,
        "answer": ""
    }
    
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    
    # 确保 _test_app_graph 在这里已经不是 None
    if _test_app_graph is None:
        print("⚠️ _test_app_graph 未初始化，正在尝试初始化...")
        _test_app_graph = await compile_app_graph() # 调用并赋值
        if _test_app_graph is None: # 再次检查以防万一
            raise RuntimeError("Failed to compile _test_app_graph in test environment.")

    final_state = await _test_app_graph.ainvoke(initial_state, config) # 使用局部变量

    print(f"🎯 Intent: {final_state.get('intent')}")
    if final_state.get('order_data'):
        order_sn = final_state['order_data'].get('order_sn') or final_state['order_data'].get('sn') or "未知"
        status = final_state['order_data'].get('status') or "未知"
        print(f"📦 Order Found: {order_sn} (Status: {status})")
    else:
        print(f"📦 Order Found: None")
    
    print(f"🤖 Answer: {final_state['answer']}")
    return final_state

async def run_test_scenarios():
    print("🚀 开始 v2.0 集成测试方案...")
    
    global _test_app_graph # 声明会修改全局变量

    print("🔧 正在初始化数据库...")
    await init_db()
    print("🔧 正在编译 LangGraph 图...")
    _test_app_graph = await compile_app_graph() # 将编译结果赋给局部变量
    # 再次检查，以防 compile_app_graph 内部出问题
    if _test_app_graph is None:
        raise RuntimeError("Failed to compile LangGraph in test setup.")

    # --- 场景 1: 用户 1 查询自己的订单 ---
    # 预期：查到 SN20240001 的详情
    await simulate_agent_call(
        query="帮我查下订单 SN20240001 的状态", 
        user_id=1, 
        thread_id="session_user_1"
    )

    # --- 场景 2: 用户 2 恶意查询用户 1 的订单 ---
    # 预期：识别为 ORDER 意图，但在 SQL 过滤后返回“未找到”，保护隐私
    await simulate_agent_call(
        query="我想看下 SN20240001 的订单详情", 
        user_id=2, 
        thread_id="session_user_2"
    )

    # --- 场景 3: 政策咨询逻辑回归 (v1 功能) ---
    # 预期：识别为 POLICY 意图，从向量库检索回答
    await simulate_agent_call(
        query="内衣拆封了可以退吗？", 
        user_id=1, 
        thread_id="session_user_1"
    )

    # --- 场景 4: 多轮对话记忆测试 (Redis Checkpointer) ---
    # 第一轮：查单
    await simulate_agent_call(
        query="我的单子 SN20240001 到了吗？", 
        user_id=1, 
        thread_id="memory_test_001"
    )
    # 第二轮：不带单号，模糊询问
    # 预期：Agent 应该记得我们在讨论刚才那个单子 (需要在 generate node 中引用 context)
    await simulate_agent_call(
        query="它是寄到哪里的？", 
        user_id=1, 
        thread_id="memory_test_001"
    )

if __name__ == "__main__":
    # 确保 Redis 和 DB 已启动
    asyncio.run(run_test_scenarios())