# scripts/etl_policy_v2.py
import asyncio
import argparse
import os
import sys
import glob  # 引入 glob 用于批量查找文件

sys.path.append(os.getcwd())

# 引入 PDF 加载器
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import delete
from tenacity import retry, stop_after_attempt, wait_exponential # 引入重试库
from pydantic import SecretStr
from app.core.config import settings
from app.core.database import async_session_maker
from app.models.knowledge import KnowledgeChunk

# ================= 配置区 =================
BATCH_SIZE = 50  # 每次向 Embedding API 发送的片段数量（防止 API 超时）
DB_BATCH_SIZE = 100 # 每次向数据库写入的条数
# =========================================

POLICY_VERSION = os.getenv("POLICY_VERSION", "dev")
POLICY_DOCUMENT_ID = os.getenv("POLICY_DOCUMENT_ID", "")

embeddings_model = OpenAIEmbeddings(
    base_url=settings.OPENAI_BASE_URL,
    api_key=SecretStr(settings.OPENAI_API_KEY),
    model=settings.EMBEDDING_MODEL,
    check_embedding_ctx_length=False
)

def get_loader(file_path: str):
    """根据文件后缀自动选择加载器"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext in [".md", ".txt"]:
        return TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def embed_with_retry(texts):
    """带重试机制的 Embedding 调用，防止网络抖动"""
    return await embeddings_model.aembed_documents(texts)

async def process_file(file_path: str, source_name: str, version: str = "", document_id: str = ""):
    print(f"🚀 [Start] 处理文件: {source_name}")
    document_id = document_id or POLICY_DOCUMENT_ID or os.path.splitext(source_name)[0]
    version = version or POLICY_VERSION
    
    try:
        # --- Step 1: 加载 ---
        loader = get_loader(file_path)
        # load() 是同步的，如果文件巨大建议用 lazy_load()，这里简单起见用 load
        docs = loader.load()
        
        # --- Step 2: 切片 ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        split_docs = text_splitter.split_documents(docs)
        total_chunks = len(split_docs)
        print(f"  📄 切分完成: {total_chunks} 个片段")

        if total_chunks == 0:
            print("  ⚠️ 警告: 文件为空或无法读取")
            return

        staged_chunk_count = 0
        async with async_session_maker() as session:
            # --- Step 3.1: 幂等清理 ---
            # 同一 source 视为同一知识文档，重跑时先清理旧片段。
            # 新片段写入 document_id/version 元数据，回答引用携带版本号。
            print(f"  broom 清理旧数据 (document_id={document_id}, version={version})...")
            await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.source == source_name))
            # Keep the prior document visible unless every replacement chunk
            # has been embedded and staged successfully.
            await session.flush()

            # --- Step 3.2: 分批处理 (Batch Processing) ---
            # 这是规避 OOM 和 API 超时的关键！
            for i in range(0, total_chunks, BATCH_SIZE):
                batch_docs = split_docs[i : i + BATCH_SIZE]
                
                # 🛠️ [修复核心]：提取文本的同时，过滤掉空字符串和纯空白字符
                batch_texts = []
                valid_indices = [] # 记录有效文本对应的原始索引，以便对齐 Metadata
                
                for idx, doc in enumerate(batch_docs):
                    cleaned_text = doc.page_content.strip() # 去除首尾空格
                    if cleaned_text:  # 只有非空文本才处理
                        batch_texts.append(cleaned_text) # 注意：这里用清洗后的文本还是原文本视需求而定，通常清洗后的更好
                        valid_indices.append(idx)

                # 如果这一批全是空行，直接跳过，不然 API 会报错
                if not batch_texts:
                    print(f"  ⚠️ 跳过空白批次 {i}")
                    continue

                # 提取对应的 MetaData (只取有效的)
                batch_metas = []
                for idx in valid_indices:
                    doc = batch_docs[idx]
                    page = doc.metadata.get("page", 0) + 1 
                    # chunk_index 依然基于全局的 i + idx
                    batch_metas.append({"page": page, "chunk_index": i + idx})

                print(f"  🧠 Embedding 批次 {i // BATCH_SIZE + 1} (有效片段: {len(batch_texts)})...")
                
                # 调用带重试的 Embedding
                vectors = await embed_with_retry(batch_texts)

                # 组装对象
                new_chunks = []
                for j, text in enumerate(batch_texts):
                    chunk = KnowledgeChunk(
                        content=text,
                        embedding=vectors[j],
                        source=source_name,
                        meta_data={**batch_metas[j], "document_id": document_id, "version": version}
                    )
                    new_chunks.append(chunk)
                
                # 写入数据库
                session.add_all(new_chunks)
                await session.flush()
                staged_chunk_count += len(new_chunks)

            # Deletion and every replacement chunk become visible atomically.
            # Any embedding/database failure exits the session without a commit,
            # preserving the previously published policy version.
            if staged_chunk_count == 0:
                raise ValueError("Policy replacement contains no non-empty chunks")
            await session.commit()
                
        print(f"✅ [Done] {source_name} 处理完毕")

    except Exception as e:
        print(f"❌ [Error] 处理文件 {file_path} 失败: {str(e)}")
        # 生产环境中这里应该记录到 error.log，而不是仅仅 print

async def main():
    parser = argparse.ArgumentParser(description="Versioned policy knowledge ETL")
    parser.add_argument("--version", default=POLICY_VERSION, help="Document version stored in chunk metadata")
    parser.add_argument("--document-id", default=POLICY_DOCUMENT_ID, help="Stable document id (default: file name)")
    args = parser.parse_args()

    # 1. 自动扫描 data 目录下的所有 PDF 和 MD 文件
    # 假设你的文件都在项目根目录的 data 文件夹下
    base_dir = "data"
    
    # 查找 pdf, md, txt
    all_files = []
    for ext in ["*.pdf", "*.md", "*.txt"]:
        all_files.extend(glob.glob(os.path.join(base_dir, ext)))

    print(f"📂 扫描到 {len(all_files)} 个文件待处理...")

    for file_path in all_files:
        # 自动生成 source_name，例如 "data/policy.pdf" -> "policy.pdf"
        source_name = os.path.basename(file_path)
        
        await process_file(file_path, source_name, version=args.version, document_id=args.document_id)

if __name__ == "__main__":
    # Windows 下如果报错 EventLoop 相关问题，可以解开下面这行的注释
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
