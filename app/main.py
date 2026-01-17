# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.chat import router as chat_router
from app.core.config import settings
from app.core.database import init_db
from app.graph.workflow import compile_app_graph
import app.graph.workflow as workflow_module

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="3.0.0",  #  更新版本号
    description="读写·退货受理专员 (The Writer) - 支持订单查询、政策咨询、退货申请"  # 更新描述
)

# 1. 配置跨域 (允许前端调用)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请改为具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 注册路由
app.include_router(chat_router, prefix=settings.API_V1_STR, tags=["Chat"])

@app.on_event("startup")
async def on_startup():
    print(" Starting up...")
    await init_db()
    workflow_module.app_graph = await compile_app_graph()
    print("Infrastructure is ready.")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": "v3.0",
        "features": ["订单查询", "政策咨询", "退货申请"]
    }