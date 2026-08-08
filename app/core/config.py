# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict, NoDecode
from pydantic import PostgresDsn, RedisDsn, computed_field, field_validator
from typing import Annotated, List

class Settings(BaseSettings):
    PROJECT_NAME: str
    API_V1_STR: str

    # Database
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: int

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        # 构建异步连接字符串: postgresql+asyncpg://...
        return str(PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_SERVER,
            port=self.POSTGRES_PORT,
            path=self.POSTGRES_DB,
        ))

    # Redis
    REDIS_HOST: str
    REDIS_PORT: int

    @computed_field
    @property
    def REDIS_URL(self) -> str:
        return str(RedisDsn.build(
            scheme="redis",
            host=self.REDIS_HOST,
            port=self.REDIS_PORT,
        ))

    # LLM (Qwen)
    OPENAI_BASE_URL: str
    OPENAI_API_KEY: str
    LLM_MODEL: str = "qwen-plus"
    EMBEDDING_MODEL: str = "text-embedding-v3"
    EMBEDDING_DIM: int = 1024
    # Java commerce anti-corruption API. This service never connects to its database.
    COMMERCE_API_BASE_URL: str = ""
    COMMERCE_API_TIMEOUT_SECONDS: float = 8.0
    # LLM call budget. Failures should degrade to order cards, FAQ search, or
    # handoff rather than blocking the commerce checkout path.
    LLM_TIMEOUT_SECONDS: float = 60.0
    # RabbitMQ domain-event inbox. Empty disables the standalone consumer.
    RABBITMQ_URL: str = ""
    COMMERCE_EVENT_QUEUE: str = "agent.runtime.events"
    # Shared only with the BFF; when set, direct browser access to Agent APIs is rejected.
    BFF_SHARED_SECRET: str = ""
    # Comma-separated explicit origins. Wildcards cannot be used with credentials.
    CORS_ORIGINS: Annotated[List[str], NoDecode] = ["http://localhost:3000"]
    # Per-user fixed-window rate limit on Agent chat endpoints.
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60

    # 风控阈值配置
    HIGH_RISK_REFUND_AMOUNT: float = 2000.0  # 高风险退款金额阈值
    MEDIUM_RISK_REFUND_AMOUNT: float = 500.0  # 中风险退款金额阈值
    
    # WebSocket 配置
    WEBSOCKET_HEARTBEAT_INTERVAL: int = 30  # 心跳间隔（秒）
    WEBSOCKET_RECONNECT_TIMEOUT: int = 60  # 重连超时（秒）
    
    # 轮询配置
    STATUS_POLLING_INTERVAL: int = 3  # 状态轮询间隔（秒）
    
    # Security configuration. Generate a unique secret for every deployed environment.
    SECRET_KEY: str 
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    model_config = SettingsConfigDict(
        env_file=".env", 
        extra="ignore"
    )

settings = Settings()
