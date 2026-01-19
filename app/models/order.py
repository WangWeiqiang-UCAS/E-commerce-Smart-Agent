# app/models/order.py
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict
from sqlalchemy import Column, JSON, String, text, Numeric
from sqlmodel import SQLModel, Field, Relationship

# 1. 使用 Enum 管理状态
class OrderStatus(str, Enum):
    PENDING = "PENDING"
    PAID = "PAID"
    SHIPPED = "SHIPPED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"

# 2. 订单模型（User 从 user.py 导入）
class Order(SQLModel, table=True):
    __tablename__ = "orders"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    order_sn: str = Field(unique=True, index=True, max_length=32)
    
    # 关联用户
    user_id:  int = Field(foreign_key="users.id", ondelete="RESTRICT")
    user: "User" = Relationship(back_populates="orders")  # 延迟导入
    
    status: OrderStatus = Field(
        default=OrderStatus.PENDING, 
        sa_column=Column(String, index=True, nullable=False)
    )
    
    total_amount: float = Field(sa_column=Column(Numeric(precision=10, scale=2)))
    items: List[Dict] = Field(default=[], sa_column=Column(JSON))
    
    tracking_number: Optional[str] = Field(default=None, index=True)
    shipping_address: str = Field(description="下单时的详细地址快照")
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")}
    )
    
    updated_at:  datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column_kwargs={
            "server_default": text("CURRENT_TIMESTAMP"),
            "onupdate": text("CURRENT_TIMESTAMP")
        }
    )

    class Config:
        use_enum_values = True