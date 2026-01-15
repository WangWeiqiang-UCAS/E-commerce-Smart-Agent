# app/models/knowledge.py
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlmodel import SQLModel, Field, Column, JSON
from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, text
from pydantic import ConfigDict

# 引入配置
from app.core.config import settings 

class KnowledgeChunk(SQLModel, table=True):
    __tablename__ = "knowledge_chunks" #type: ignore
    
    # 动态获取维度用于 HNSW 索引参数
    __table_args__ = (
        Index(
            "ix_knowledge_chunks_embedding",
            text("embedding vector_cosine_ops"),
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64}
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    content: str = Field(index=False)

    # 从 Config 读取维度
    # 注意：SQLAlchemy 在定义类时就会执行 Vector(dim)，
    # 所以 settings.EMBEDDING_DIM 必须在此时就是可用的整数。
    embedding: List[float] = Field(
        sa_column=Column(Vector(settings.EMBEDDING_DIM)) 
    )

    source: str = Field(index=True)
    meta_data: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    is_active: bool = Field(default=True, index=True)

    created_at: datetime = Field(
        # replace(tzinfo=None) 会移除时区信息，但保留 UTC 的时间数值
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

    model_config = {"arbitrary_types_allowed": True}
        