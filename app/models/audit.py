# app/models/audit.py
"""
v4.0 新增：审计日志模型
记录人工审核流程的完整上下文
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, text, JSON, Text
from sqlmodel import SQLModel, Field


class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AuditAction(str, Enum):
    """审核动作"""
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"  
    PENDING = "PENDING"   


class AuditLog(SQLModel, table=True):
    """审计日志表 - 记录所有需要人工介入的决策"""
    __tablename__ = "audit_logs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # 会话标识
    thread_id: str = Field(index=True, max_length=128, description="会话ID")
    
    # 关联的订单/申请ID
    order_id: Optional[int] = Field(default=None, index=True)
    refund_application_id: Optional[int] = Field(default=None, index=True)
    
    # 用户信息
    user_id: int = Field(index=True, description="发起用户ID")
    
    # 触发原因
    trigger_reason: str = Field(
        sa_column=Column(Text, nullable=False),
        description="触发人工审核的原因"
    )
    
    # 风险等级
    risk_level: RiskLevel = Field(
        default=RiskLevel.LOW,
        sa_column=Column(String, index=True, nullable=False)
    )
    
    # 审核状态
    action: AuditAction = Field(
        default=AuditAction.PENDING,
        sa_column=Column(String, index=True, nullable=False)
    )
    
    # 管理员信息
    admin_id: Optional[int] = Field(default=None, index=True, description="审核管理员ID")
    admin_comment: Optional[str] = Field(
        default=None,
        sa_column=Column(Text),
        description="管理员备注"
    )
    
    # 上下文快照 (保存触发时的完整对话历史和订单详情)
    context_snapshot:  Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False),
        description="触发时的上下文快照"
    )
    
    # 决策结果元数据
    decision_metadata:  Optional[Dict[str, Any]] = Field(
        default={},
        sa_column=Column(JSON),
        description="决策相关的元数据"
    )
    
    # 时间戳
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
        description="创建时间（触发审核时间）"
    )
    
    reviewed_at: Optional[datetime] = Field(
        default=None,
        description="审核完成时间"
    )
    
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column_kwargs={
            "server_default": text("CURRENT_TIMESTAMP"),
            "onupdate": text("CURRENT_TIMESTAMP")
        }
    )
    
    class Config:
        use_enum_values = True