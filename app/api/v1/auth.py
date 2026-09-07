"""Internal support-operator authentication only.

Customer registration and membership are owned by Java commerce. This module is
used exclusively by the protected agent-desk workspace.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlmodel import select

from app.core.database import async_session_maker
from app.core.security import create_access_token, get_admin_user_id, require_bff_secret
from app.models.support_operator import SupportOperator

router = APIRouter()


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    operator_id: int
    username: str
    full_name: str
    is_admin: bool


class OperatorInfoResponse(BaseModel):
    operator_id: int
    username: str
    email: str
    full_name: str
    is_admin: bool
    created_at: str


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, _: None = Depends(require_bff_secret)):
    """Authenticate an internal support operator, never a commerce member."""
    async with async_session_maker() as session:
        operator = (await session.execute(
            select(SupportOperator).where(SupportOperator.username == request.username)
        )).scalar_one_or_none()
        if not operator or not operator.is_active or not operator.verify_password(request.password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid operator credentials")
        if not operator.is_admin:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Support desk administrator privileges required")
        return TokenResponse(
            access_token=create_access_token(user_id=operator.id, is_admin=True),
            operator_id=operator.id,
            username=operator.username,
            full_name=operator.full_name,
            is_admin=True,
        )


@router.get("/me", response_model=OperatorInfoResponse)
async def current_operator(operator_id: int = Depends(get_admin_user_id)):
    async with async_session_maker() as session:
        operator = await session.get(SupportOperator, operator_id)
        if not operator:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operator not found")
        return OperatorInfoResponse(
            operator_id=operator.id,
            username=operator.username,
            email=operator.email,
            full_name=operator.full_name,
            is_admin=operator.is_admin,
            created_at=operator.created_at.isoformat(),
        )
