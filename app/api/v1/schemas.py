# app/api/v1/schemas.py
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    # 用户的问题
    question: str = Field(
        ..., json_schema_extra={"example": "内衣拆封了可以退吗？"}
    )
    
    # 会话 ID，用于后续追踪对话上下文 (v1.0 暂不强制，但预留)
    thread_id: str = Field(
        "default_thread",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
        json_schema_extra={"example": "support_session_001"},
    )

class ChatResponse(BaseModel):
    # 非流式模式下的返回结构
    answer: str
