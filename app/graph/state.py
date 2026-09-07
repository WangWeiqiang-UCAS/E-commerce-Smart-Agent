# app/graph/state.py
from typing import TypedDict, List, Optional, Annotated, Dict, Any
import operator

class AgentState(TypedDict):
    # 基础信息
    question: str
    user_id: int  
    
    # 意图标签:  "GUIDE" | "ORDER" | "POLICY" | "REFUND" | "OTHER"
    intent: Optional[str]
    
    # 历史记录 (用于多轮对话)
    history:  Annotated[List[dict], operator.add]
    
    # 检索到的知识 
    context:  List[str]
    
    # 查到的订单数据 
    order_data: Optional[dict]
    order_detail_loaded: bool
    logistics_data: Optional[dict]

    # 导购子图：商品搜索结果（来自 Java 只读 Tool）
    products: Optional[List[dict]]
    product_detail: Optional[dict]

    # 售后子图：售后进度（来自 Java 只读 Tool）
    after_sale_data: Optional[dict]

    # 售后子图：写操作预览交接载荷（仅在用户显式确认后产生）
    handoff: Optional[dict]
    
    # v4.0 新增：会话 ID
    thread_id: str
    
    # v4.0 新增：审核状态
    audit_required: bool  # 是否需要人工审核
    audit_log_id: Optional[int]  # 审计日志ID
    
    # v4.0 新增：结构化消息列表
    messages:  Annotated[List[Dict[str, Any]], operator.add]
    
    # v3.0 保留：退货流程状态
    refund_flow_active: Optional[bool]
    refund_order_sn: Optional[str]
    refund_step: Optional[str]
    
    # 最终回复
    answer: str
