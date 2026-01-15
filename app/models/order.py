from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict
from sqlalchemy import Column, JSON, String, text, Numeric
from sqlmodel import SQLModel, Field, Relationship

# 1. 使用 Enum 管理状态，防止硬编码错误
class OrderStatus(str, Enum):
    PENDING = "PENDING"
    PAID = "PAID"
    SHIPPED = "SHIPPED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"

# 2. 定义用户模型以支持关系
class User(SQLModel, table=True):
    __tablename__ = "users"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True, max_length=50) 
    email: str = Field(unique=True, index=True)
    full_name: str
    
    # 一个用户可以有多个订单
    orders: List["Order"] = Relationship(back_populates="user")

# 3. 定义订单模型
class Order(SQLModel, table=True):
    __tablename__ = "orders"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    # 唯一约束，防止重复提交
    order_sn: str = Field(unique=True, index=True, max_length=32)
    
    # 增加级联删除说明，通常订单不随用户删除而删除，而是设为 Restricted 或 Set Null
    user_id: int = Field(foreign_key="users.id", ondelete="RESTRICT")
    user: User = Relationship(back_populates="orders")
    
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
    
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column_kwargs={
            "server_default": text("CURRENT_TIMESTAMP"),
            "onupdate": text("CURRENT_TIMESTAMP")
        }
    )

    class Config:
        # 允许使用 Enum
        use_enum_values = True