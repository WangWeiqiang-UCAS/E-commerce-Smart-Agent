# scripts/test_refund_subgraph.py
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph.refund_subgraph import refund_subgraph


async def test_refund_subgraph():
    """测试退货子流程图"""
    
    print("=" * 60)
    print("🧪 测试退货子流程图")
    print("=" * 60)
    
    # ========== 场景 1: 完整流程（一次性提供所有信息） ==========
    print("\n📋 场景 1: 用户一次性提供订单号和原因")
    
    initial_state = {
        "user_id": 1,
        "question": "我要退货，订单号是 SN20240003，尺码太大了",
        "order_sn":  None,
        "order_id":  None,
        "eligibility_check": None,
        "reason_detail": None,
        "reason_category": None,
        "current_step": "extract_order",
        "needs_user_input": False,
        "response": ""
    }
    
    result = await refund_subgraph.ainvoke(initial_state)
    print(f"\n最终回复：\n{result.get('response', '无回复')}")
    print(f"是否需要用户输入:  {result.get('needs_user_input', False)}")
    
    # ========== 场景 2: 分步流程（模拟多轮对话） ==========
    print("\n" + "=" * 60)
    print("📋 场景 2: 分步流程（模拟多轮对话）")
    print("=" * 60)
    
    # 第一轮：用户只说"我要退货"
    print("\n👤 用户:  我要退货")
    state = {
        "user_id":  1,
        "question":  "我要退货",
        "order_sn":  None,
        "order_id":  None,
        "eligibility_check": None,
        "reason_detail": None,
        "reason_category": None,
        "current_step": "extract_order",
        "needs_user_input": False,
        "response": ""
    }
    
    result = await refund_subgraph.ainvoke(state)
    print(f"🤖 Agent:  {result.get('response', '无回复')}")
    
    # 第二轮：用户提供订单号
    if result.get("needs_user_input"):
        print("\n👤 用户: SN20240004")
        state = {
            "user_id": 1,
            "question": "SN20240004",
            "order_sn": None,
            "order_id":  None,
            "eligibility_check": None,
            "reason_detail": None,
            "reason_category": None,
            "current_step": "extract_order",
            "needs_user_input": False,
            "response": ""
        }
        
        result = await refund_subgraph.ainvoke(state)
        print(f"🤖 Agent: {result.get('response', '无回复')}")
    
    # 第三轮：用户提供退货原因
    if result.get("needs_user_input") and result.get("current_step") == "collect_reason":
        print("\n👤 用户: 鞋码偏大，穿着不舒服")
        state = {
            "user_id": 1,
            "question": "鞋码偏大，穿着不舒服",
            "order_sn":  result.get("order_sn"),
            "order_id": result.get("order_id"),
            "eligibility_check": result.get("eligibility_check"),
            "reason_detail": None,
            "reason_category":  None,
            "current_step": "collect_reason",
            "needs_user_input": False,
            "response": ""
        }
        
        result = await refund_subgraph.ainvoke(state)
        print(f"🤖 Agent: {result.get('response', '无回复')}")
    
    # ========== 场景 3: 不符合退货条件 ==========
    print("\n" + "=" * 60)
    print("📋 场景 3: 订单不符合退货条件（内衣）")
    print("=" * 60)
    
    state = {
        "user_id":  1,
        "question":  "我要退 SN20240001",
        "order_sn": None,
        "order_id": None,
        "eligibility_check": None,
        "reason_detail": None,
        "reason_category": None,
        "current_step": "extract_order",
        "needs_user_input": False,
        "response": ""
    }
    
    result = await refund_subgraph.ainvoke(state)
    print(f"🤖 Agent:  {result.get('response', '无回复')}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio. run(test_refund_subgraph())