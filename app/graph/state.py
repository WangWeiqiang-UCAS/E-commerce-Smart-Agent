# app/graph/state.py
from typing import TypedDict, List, Optional, Annotated
import operator

class AgentState(TypedDict):
    # 基础信息
    question: str
    user_id: int  
    
    # 意图标签: "POLICY" 或 "ORDER" 或 "OTHER"
    intent: Optional[str]
    
    # 历史记录 (用于多轮对话)
    history: Annotated[List[dict], operator.add]
    
    # 检索到的知识 (v1)
    context: List[str]
    
    # 查到的订单数据 (v2)
    order_data: Optional[dict]
    
    # 最终回复
    answer: str