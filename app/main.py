# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.chat import router as chat_router
from app.api.v1.auth import router as auth_router  # v4.0 新增
from app.core.config import settings
from app.core.tracing import TraceIdMiddleware
from app.core.database import init_db
from app.graph.workflow import compile_app_graph
import app.graph.workflow as workflow_module
from app.api.v1.admin import router as admin_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="4.0.0",
    description="全栈·沉浸式人机协作系统 (The Immersive System) - v4.0"
)

# 1. 配置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TraceIdMiddleware)

# 2. 注册路由
app.include_router(auth_router, prefix=settings.API_V1_STR, tags=["Auth"])  # v4.0 新增
app.include_router(chat_router, prefix=settings.API_V1_STR, tags=["Chat"])
app.include_router(admin_router, prefix=settings.API_V1_STR, tags=["Admin"])


@app.on_event("startup")
async def on_startup():
    print(" Starting E-commerce Smart Agent v4.0...")
    await init_db()
    workflow_module.app_graph = await compile_app_graph()
    print(" Infrastructure is ready.")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": "v4.0",
        "features": [
            "用户登录认证", 
            "多租户数据隔离",
            "导购/订单/政策/售后子图",
            "订单查询",
            "政策咨询",
            "退款预览确认交接",
            "人工会话工作台",
            "实时状态同步",
            "管理端只读会话审计"
        ]
    }
