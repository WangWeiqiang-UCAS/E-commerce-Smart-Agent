# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.chat import router as chat_router
from app.core.config import settings
from app.core.database import init_db
from app.graph.workflow import compile_app_graph # 导入编译函数
import app.graph.workflow as workflow_module # 导入整个模块，以便在 chat.py 中也能访问

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="只读·政策咨询专家 (The Knowledge Base)"
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
    print("🌊 Starting up...")
    await init_db()
    # 调用编译函数，并将结果赋值给模块中的 app_graph 变量
    workflow_module.app_graph = await compile_app_graph() # 将编译结果赋给模块变量
    print("✅ Infrastructure is ready.")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "v1.0"}