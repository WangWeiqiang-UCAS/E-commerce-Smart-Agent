# app/models/user.py
"""
用户模型 - 支持真实登录认证
"""
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import text
from sqlmodel import SQLModel, Field, Relationship
from passlib.context import CryptContext

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(SQLModel, table=True):
    """用户表"""
    __tablename__ = "users"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # 用户名（登录凭证）
    username: str = Field(index=True, unique=True, max_length=50, description="用户名")
    
    # 密码哈希
    password_hash:  str = Field(max_length=255, description="密码哈希")
    
    # 用户信息
    email: str = Field(unique=True, index=True, description="邮箱")
    full_name: str = Field(max_length=100, description="真实姓名")
    phone: Optional[str] = Field(default=None, max_length=20, description="手机号")
    
    # 角色权限
    is_admin: bool = Field(default=False, description="是否为管理员")
    is_active: bool = Field(default=True, description="账号是否激活")
    
    # 时间戳
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
    
    # 关系
    orders: List["Order"] = Relationship(back_populates="user")
    
    @staticmethod
    def hash_password(password: str) -> str:
        """密码加密"""
        return pwd_context.hash(password)
    
    def verify_password(self, password: str) -> bool:
        """验证密码"""
        return pwd_context.verify(password, self.password_hash)
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "alice",
                "email": "alice@example.com",
                "full_name": "Alice Wang"
            }
        }