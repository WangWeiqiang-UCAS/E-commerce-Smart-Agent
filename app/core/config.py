# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import PostgresDsn, RedisDsn, computed_field

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
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

    # === 安全配置 ===
    # 建议生产环境使用: openssl rand -hex 32 生成
    SECRET_KEY: str 
    ALGORITHM: str 
    # Token 有效期（分钟），默认 1 天
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    # 允许 Pydantic 读取 .env 文件
    model_config = SettingsConfigDict(
        env_file=".env", 
        extra="ignore" # 忽略多余的环境变量
    )

    # Celery 配置
    CELERY_BROKER_URL: str = ""  # 默认使用 REDIS_URL
    CELERY_RESULT_BACKEND: str = ""  # 默认使用 REDIS_URL
    
    @computed_field
    @property
    def CELERY_BROKER(self) -> str:
        """Celery Broker URL"""
        return self.CELERY_BROKER_URL or self.REDIS_URL
    
    @computed_field
    @property
    def CELERY_BACKEND(self) -> str:
        """Celery Result Backend URL"""
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL
    
    # 风控阈值配置
    HIGH_RISK_REFUND_AMOUNT: float = 2000.0  # 高风险退款金额阈值
    MEDIUM_RISK_REFUND_AMOUNT: float = 500.0  # 中风险退款金额阈值
    
    # WebSocket 配置
    WEBSOCKET_HEARTBEAT_INTERVAL: int = 30  # 心跳间隔（秒）
    WEBSOCKET_RECONNECT_TIMEOUT: int = 60  # 重连超时（秒）
    
    # 轮询配置
    STATUS_POLLING_INTERVAL: int = 3  # 状态轮询间隔（秒）
    
    # 安全配置
    SECRET_KEY: str 
    ALGORITHM: str 
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    model_config = SettingsConfigDict(
        env_file=".env", 
        extra="ignore"
    )

settings = Settings()