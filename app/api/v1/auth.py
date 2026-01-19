# app/api/v1/auth.py
"""
认证 API - 登录、注册
"""
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from app.core.security import create_access_token, get_current_user_id
from app.core.database import async_session_maker
from app.models.user import User
from sqlmodel import select

router = APIRouter()


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    password: str = Field(..., min_length=6, description="密码")


class RegisterRequest(BaseModel):
    """注册请求"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    password: str = Field(..., min_length=6, description="密码")
    email: EmailStr = Field(..., description="邮箱")
    full_name: str = Field(..., min_length=2, max_length=100, description="真实姓名")
    phone: Optional[str] = Field(default=None, description="手机号")


class TokenResponse(BaseModel):
    """Token 响应"""
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    full_name: str
    is_admin: bool


class UserInfoResponse(BaseModel):
    """用户信息响应"""
    user_id: int
    username: str
    email: str
    full_name:  str
    phone: Optional[str]
    is_admin: bool
    created_at: str


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    用户登录
    
    验证用户名和密码，返回 JWT Token
    """
    async with async_session_maker() as session:
        # 查询用户
        result = await session.execute(
            select(User).where(User.username == request.username)
        )
        user = result.scalar_one_or_none()
        
        # 验证用户存在
        if not user: 
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 验证账号激活
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="账号已被禁用，请联系管理员"
            )
        
        # 验证密码
        if not user.verify_password(request.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 生成 Token
        token = create_access_token(user_id=user.id, is_admin=user.is_admin)
        
        return TokenResponse(
            access_token=token,
            user_id=user.id,
            username=user.username,
            full_name=user.full_name,
            is_admin=user.is_admin
        )


@router.post("/register", response_model=TokenResponse)
async def register(request:  RegisterRequest):
    """
    用户注册
    
    创建新用户并返回 JWT Token
    """
    async with async_session_maker() as session:
        # 检查用户名是否已存在
        result = await session.execute(
            select(User).where(User.username == request.username)
        )
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已存在"
            )
        
        # 检查邮箱是否已存在
        result = await session.execute(
            select(User).where(User.email == request.email)
        )
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱已被注册"
            )
        
        # 创建用户
        user = User(
            username=request.username,
            password_hash=User.hash_password(request.password),
            email=request.email,
            full_name=request.full_name,
            phone=request.phone,
            is_admin=False,
            is_active=True
        )
        
        session.add(user)
        await session.commit()
        await session.refresh(user)
        
        # 生成 Token
        token = create_access_token(user_id=user.id, is_admin=user.is_admin)
        
        return TokenResponse(
            access_token=token,
            user_id=user.id,
            username=user.username,
            full_name=user.full_name,
            is_admin=user.is_admin
        )


@router.get("/me", response_model=UserInfoResponse)
async def get_current_user_info(current_user_id: int = Depends(get_current_user_id)):
    """
    获取当前登录用户信息
    """
    from app.core.security import get_current_user_id
    
    async with async_session_maker() as session:
        user = await session.get(User, current_user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        return UserInfoResponse(
            user_id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            phone=user.phone,
            is_admin=user.is_admin,
            created_at=user.created_at.isoformat()
        )