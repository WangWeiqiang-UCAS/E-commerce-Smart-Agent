import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# 1. 引入配置和模型
from app.core.config import settings
from sqlmodel import SQLModel
# 导入所有模型以确保被注册
from app.models.knowledge import KnowledgeChunk 
from app.models.audit import AuditLog
from app.models.message import MessageCard
from app.models.user import User
from app.models.inbox import InboxEvent
# ==========================================

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata

# =========================================================
# 强制处理数据库 URL 协议
# =========================================================
def get_url():
    url = settings.DATABASE_URL
    # 如果配置的是标准 postgresql://，强制替换为异步驱动 postgresql+asyncpg://
    # 这样既兼容了同步代码(用 psycopg2)，也兼容了这里的异步迁移
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url

# 将处理后的 URL 注入配置
config.set_main_option("sqlalchemy.url", get_url())


def run_migrations_offline() -> None:
    """离线模式"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # 💡 忽略 vector 类型检查，防止 alembic 在 --autogenerate 时因为不认识 vector 而删除它
        # 仅在某些旧版本 alembic 需要，保留以防万一
        compare_type=True, 
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """执行迁移"""
    context.configure(
        connection=connection, 
        target_metadata=target_metadata,
        # 💡 启用类型比较，否则 Alembic 可能会忽略字段类型的变化（如 varchar 长度变化）
        compare_type=True
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """在线模式"""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
