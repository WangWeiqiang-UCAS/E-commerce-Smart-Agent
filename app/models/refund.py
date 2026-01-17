# app/models/refund.py
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from sqlalchemy import Column, String, text, Numeric, Text
from sqlmodel import SQLModel, Field, Relationship

# 1. 退货申请状态枚举
class RefundStatus(str, Enum):
    """退货申请状态"""
    PENDING = "PENDING"           # 待审核
    APPROVED = "APPROVED"         # 已批准
    REJECTED = "REJECTED"         # 已拒绝
    COMPLETED = "COMPLETED"       # 已完成（货物已收到）
    CANCELLED = "CANCELLED"       # 用户取消

# 2. 退货原因枚举（可选，用于数据分析）
class RefundReason(str, Enum):
    """退货原因分类"""
    QUALITY_ISSUE = "QUALITY_ISSUE"           # 质量问题
    SIZE_NOT_FIT = "SIZE_NOT_FIT"             # 尺码不合适
    NOT_AS_DESCRIBED = "NOT_AS_DESCRIBED"     # 与描述不符
    CHANGED_MIND = "CHANGED_MIND"             # 不想要了
    OTHER = "OTHER"                           # 其他原因

# 3. 退货申请表
class RefundApplication(SQLModel, table=True):
    """退货申请表"""
    __tablename__ = "refund_applications"
    
    id:  Optional[int] = Field(default=None, primary_key=True)
    
    # 关联订单（外键）
    order_id: int = Field(foreign_key="orders.id", index=True, ondelete="RESTRICT")
    
    # 申请用户（冗余字段，方便查询）
    user_id: int = Field(foreign_key="users.id", index=True, ondelete="RESTRICT")
    
    # 退货状态
    status: RefundStatus = Field(
        default=RefundStatus.PENDING,
        sa_column=Column(String, index=True, nullable=False)
    )
    
    # 退货原因分类
    reason_category: Optional[RefundReason] = Field(
        default=None,
        sa_column=Column(String, nullable=True)
    )
    
    # 退货原因详细描述（用户填写）
    reason_detail: str = Field(
        sa_column=Column(Text, nullable=False),
        description="用户填写的退货原因"
    )
    
    # 退款金额（可能部分退款）
    refund_amount: float = Field(
        sa_column=Column(Numeric(precision=10, scale=2)),
        description="退款金额"
    )
    
    # 审核备注（管理员填写）
    admin_note: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    
    # 审核人ID（预留字段）
    reviewed_by: Optional[int] = Field(default=None)
    
    # 审核时间
    reviewed_at:  Optional[datetime] = Field(default=None)
    
    # 创建时间
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")}
    )
    
    # 更新时间
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column_kwargs={
            "server_default": text("CURRENT_TIMESTAMP"),
            "onupdate": text("CURRENT_TIMESTAMP")
        }
    )

    class Config:
        use_enum_values = True