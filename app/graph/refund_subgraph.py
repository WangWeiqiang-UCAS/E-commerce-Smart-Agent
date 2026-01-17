# app/graph/refund_subgraph.py
"""
退货子流程图 (SubGraph)
处理完整的退货申请流程：资格校验 -> 原因收集 -> 提交申请
"""
from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.core.config import settings
from app.core.database import async_session_maker
from app.services.refund_service import RefundApplicationService, RefundEligibilityChecker, RefundReason
from app.models.order import Order
from sqlmodel import select
import re


# ==========================================
# 子图状态定义
# ==========================================

class RefundFlowState(TypedDict):
    """退货流程状态"""
    # 用户信息
    user_id: int
    question:  str  # 用户的原始问题
    
    # 流程数据
    order_sn: Optional[str]  # 订单号
    order_id: Optional[int]  # 订单ID
    eligibility_check: Optional[str]  # 资格检查结果
    reason_detail: Optional[str]  # 退货原因
    reason_category: Optional[str]  # 原因分类
    
    # 流程控制
    current_step: str  # 当前步骤:  extract_order -> check_eligibility -> collect_reason -> submit
    needs_user_input: bool  # 是否需要用户输入
    
    # 最终回复
    response: str


# ==========================================
# LLM 初始化
# ==========================================

llm = ChatOpenAI(
    base_url=settings.OPENAI_BASE_URL,
    api_key=SecretStr(settings.OPENAI_API_KEY),
    model=settings.LLM_MODEL,
    temperature=0
)


# ==========================================
# 子图节点函数
# ==========================================

async def extract_order_number(state: RefundFlowState) -> dict:
    """
    步骤 1: 提取订单号
    """
    print(f"🔍 [RefundFlow] 步骤1: 提取订单号")
    
    question = state["question"]
    
    # 方法1: 正则提取订单号
    order_sn_match = re.search(r'SN\d+', question.upper())
    
    if order_sn_match:
        order_sn = order_sn_match.group()
        print(f"   ✅ 提取到订单号:  {order_sn}")
        return {
            "order_sn":  order_sn,
            "current_step": "check_eligibility",
            "needs_user_input": False
        }
    
    # 方法2: 使用 LLM 提取（处理口语化表达）
    prompt = f"""
从用户的问题中提取订单号。订单号格式为 SN 开头 + 数字，例如 SN20240001。

用户问题：{question}

如果找到订单号，只返回订单号（如 SN20240001）。
如果没有找到，返回 "NOT_FOUND"。
"""
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    extracted = response.content.strip().upper()
    
    if extracted.startswith("SN") and extracted != "NOT_FOUND":
        print(f"   ✅ LLM 提取到订单号: {extracted}")
        return {
            "order_sn": extracted,
            "current_step": "check_eligibility",
            "needs_user_input": False
        }
    
    # 未找到订单号，需要询问用户
    print(f"   ❌ 未找到订单号，需要询问用户")
    return {
        "current_step": "extract_order",
        "needs_user_input": True,
        "response": (
            "您好，我可以帮您办理退货。\n\n"
            "请提供您的订单号（格式如：SN20240001），"
            "或者告诉我您最近购买的商品名称，我帮您查询订单。"
        )
    }


async def check_refund_eligibility(state: RefundFlowState) -> dict:
    """
    步骤 2: 检查退货资格
    """
    print(f"🔍 [RefundFlow] 步骤2: 检查退货资格")
    
    order_sn = state["order_sn"]
    user_id = state["user_id"]
    
    async with async_session_maker() as session:
        # 1. 查询订单
        stmt = select(Order).where(
            Order.order_sn == order_sn,
            Order.user_id == user_id  # 🔒 安全校验
        )
        result = await session.exec(stmt)
        order = result.first()
        
        if not order:
            print(f"   ❌ 订单不存在或无权访问")
            return {
                "current_step": "end",
                "needs_user_input": False,
                "response": f"❌ 抱歉，未找到订单 {order_sn}，或您无权访问此订单。\n\n请检查订单号是否正确。"
            }
        
        # 2. 资格检查
        is_eligible, message = await RefundEligibilityChecker.check_eligibility(
            order, session
        )
        
        if is_eligible:
            print(f"   ✅ 资格检查通过")
            # 格式化订单信息
            items_str = ", ".join([f"{item['name']}(¥{item['price']})" for item in order.items])
            
            return {
                "order_id": order.id,
                "eligibility_check": "PASS",
                "current_step": "collect_reason",
                "needs_user_input": True,
                "response": (
                    f"✅ 订单 {order_sn} 符合退货条件。\n\n"
                    f"📦 订单信息：\n"
                    f"  - 商品：{items_str}\n"
                    f"  - 金额：¥{order.total_amount}\n"
                    f"  - 状态：{order.status}\n\n"
                    f"请问您的退货原因是什么？\n"
                    f"（例如：尺码不合适、质量问题、不喜欢等）"
                )
            }
        else: 
            print(f"   ❌ 资格检查失败:  {message}")
            return {
                "eligibility_check": "FAIL",
                "current_step": "end",
                "needs_user_input": False,
                "response": (
                    f"❌ 抱歉，订单 {order_sn} 不符合退货条件。\n\n"
                    f"原因：{message}\n\n"
                    f"如有疑问，请联系客服：400-XXX-XXXX"
                )
            }


async def collect_refund_reason(state: RefundFlowState) -> dict:
    """
    步骤 3: 收集退货原因
    """
    print(f"🔍 [RefundFlow] 步骤3: 收集退货原因")
    
    question = state["question"]
    
    # 使用 LLM 提取退货原因和分类
    prompt = f"""
分析用户的退货原因，并归类。

用户描述：{question}

请返回 JSON 格式：
{{
    "reason_detail": "用户的原始描述",
    "reason_category": "分类代码"
}}

分类代码规则：
- QUALITY_ISSUE: 质量问题、坏了、破损等
- SIZE_NOT_FIT: 尺码不合适、大了、小了等
- NOT_AS_DESCRIBED: 与描述不符、颜色不对、款式不对等
- CHANGED_MIND: 不想要了、不喜欢、后悔了等
- OTHER: 其他原因

只返回 JSON，不要其他文字。
"""
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    try:
        import json
        result = json.loads(response.content)
        reason_detail = result.get("reason_detail", question)
        reason_category = result.get("reason_category", "OTHER")
    except: 
        # LLM 解析失败，使用原始输入
        reason_detail = question
        reason_category = "OTHER"
    
    print(f"   原因:  {reason_detail}")
    print(f"   分类: {reason_category}")
    
    return {
        "reason_detail": reason_detail,
        "reason_category": reason_category,
        "current_step": "submit",
        "needs_user_input": False
    }


async def submit_refund_application(state: RefundFlowState) -> dict:
    """
    步骤 4: 提交退货申请
    """
    print(f"🔍 [RefundFlow] 步骤4: 提交退货申请")
    
    order_id = state["order_id"]
    user_id = state["user_id"]
    reason_detail = state["reason_detail"]
    reason_category = state.get("reason_category")
    
    # 转换原因分类
    category = None
    if reason_category: 
        try:
            category = RefundReason(reason_category)
        except ValueError:
            category = RefundReason.OTHER
    
    async with async_session_maker() as session:
        success, message, refund_app = await RefundApplicationService.create_refund_application(
            order_id=order_id,
            user_id=user_id,
            reason_detail=reason_detail,
            reason_category=category,
            session=session
        )
        
        if success and refund_app:
            print(f"   ✅ 申请提交成功，申请ID:  {refund_app.id}")
            return {
                "current_step": "end",
                "needs_user_input": False,
                "response": (
                    f"✅ 退货申请提交成功！\n\n"
                    f"📋 申请信息：\n"
                    f"  - 申请编号：#{refund_app.id}\n"
                    f"  - 订单号：{state['order_sn']}\n"
                    f"  - 退款金额：¥{refund_app.refund_amount}\n"
                    f"  - 申请状态：{refund_app.status}（待审核）\n"
                    f"  - 退货原因：{refund_app.reason_detail}\n\n"
                    f"⏳ 后续流程：\n"
                    f"  1. 我们会在 1-2 个工作日内审核您的申请\n"
                    f"  2. 审核通过后，请将商品寄回（保持包装完好）\n"
                    f"  3. 收到退货后，我们会在 3-5 个工作日内完成退款\n\n"
                    f"💡 您可以随时回复\"查询退货进度\"了解最新状态。"
                )
            }
        else:
            print(f"   ❌ 申请提交失败:  {message}")
            return {
                "current_step": "end",
                "needs_user_input": False,
                "response": f"❌ 退货申请提交失败。\n\n原因：{message}"
            }


# ==========================================
# 路由函数
# ==========================================

def route_refund_flow(state: RefundFlowState) -> Literal["extract_order", "check_eligibility", "collect_reason", "submit", "end"]:
    """根据当前步骤路由到下一个节点"""
    current_step = state.get("current_step", "extract_order")
    print(f"🔀 [RefundFlow] 路由到:  {current_step}")
    return current_step


# ==========================================
# 构建子图
# ==========================================

def create_refund_subgraph() -> StateGraph: 
    """创建退货子流程图"""
    
    # 创建子图
    subgraph = StateGraph(RefundFlowState)
    
    # 添加节点
    subgraph.add_node("extract_order", extract_order_number)
    subgraph.add_node("check_eligibility", check_refund_eligibility)
    subgraph.add_node("collect_reason", collect_refund_reason)
    subgraph.add_node("submit", submit_refund_application)
    
    # 设置入口点
    subgraph.set_entry_point("extract_order")
    
    # 添加条件路由
    subgraph.add_conditional_edges(
        "extract_order",
        route_refund_flow,
        {
            "extract_order": END,  # 需要用户输入订单号，暂停流程
            "check_eligibility": "check_eligibility"
        }
    )
    
    subgraph.add_conditional_edges(
        "check_eligibility",
        route_refund_flow,
        {
            "collect_reason": "collect_reason",
            "end": END
        }
    )
    
    subgraph.add_conditional_edges(
        "collect_reason",
        route_refund_flow,
        {
            "submit": "submit"
        }
    )
    
    subgraph.add_conditional_edges(
        "submit",
        route_refund_flow,
        {
            "end": END
        }
    )
    
    return subgraph


# ==========================================
# 编译子图
# ==========================================

refund_subgraph = create_refund_subgraph().compile()