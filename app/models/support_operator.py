"""Internal support-workspace operators; commerce customers never live here."""
from datetime import datetime, timezone
from typing import Optional

from passlib.context import CryptContext
from sqlalchemy import text
from sqlmodel import Field, SQLModel

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SupportOperator(SQLModel, table=True):
    __tablename__ = "support_operators"

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True, max_length=50)
    password_hash: str = Field(max_length=255)
    email: str = Field(unique=True, index=True)
    full_name: str = Field(max_length=100)
    phone: Optional[str] = Field(default=None, max_length=20)
    is_admin: bool = Field(default=False)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None), sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")})
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None), sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP"), "onupdate": text("CURRENT_TIMESTAMP")})

    def verify_password(self, password: str) -> bool:
        return pwd_context.verify(password, self.password_hash)
