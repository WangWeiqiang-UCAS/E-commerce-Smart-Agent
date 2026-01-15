# test_infra.py
import pytest  # 1. 导入 pytest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sqlalchemy import text
from app.core.config import settings
from app.core.database import engine

# 2. 加上这个装饰器 (如果你没有在 pyproject.toml 配置 asyncio_mode=auto)
@pytest.mark.asyncio
async def test_llm():
    print("🤖 Testing Qwen LLM Connection...")
    # 3. 去掉 try...except，让 pytest 捕获真正的错误
    llm = ChatOpenAI(
        base_url=settings.OPENAI_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
        model=settings.LLM_MODEL
    )
    response = await llm.ainvoke("你好，请回复'Pong'。") # 注意：在 async 函数中最好用 ainvoke
    print(f"✅ LLM Response: {response.content}")
    
    # 4. 加上断言 (Assert)，这是测试的核心
    assert response.content is not None
    assert len(response.content) > 0

@pytest.mark.asyncio
async def test_embedding():
    print("\n🧠 Testing Qwen Embedding...")
    emb = OpenAIEmbeddings(
        base_url=settings.OPENAI_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
        model=settings.EMBEDDING_MODEL,
        check_embedding_ctx_length=False
    )
    # embed_query 通常是同步方法，但有些库版本可能是异步，如果是异步请加 await
    vector = emb.embed_query("测试文本") 
    print(f"✅ Embedding Success. Dimension: {len(vector)}")
    
    assert len(vector) > 0
    # 通常 embedding 维度是固定的（例如 1536 或 1024），你可以加上具体维度的检查
    # assert len(vector) == 1024 

@pytest.mark.asyncio
async def test_db():
    print("\n🗄️ Testing Database Connection...")
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        value = result.scalar()
        print(f"✅ DB Connected: {value}")
        
        assert value == 1
        
        # 检查 pgvector 扩展
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        print("✅ pgvector extension ensured.")

# 5. 如果只用 pytest 运行，下面的 main 和 if __name__ 其实可以删掉
# 但保留着也不影响