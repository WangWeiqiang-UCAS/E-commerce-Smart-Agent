# app/graph/nodes.py
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import select, text
from pydantic import SecretStr
from app.core.config import settings
from app.core.database import async_session_maker
from app.models.knowledge import KnowledgeChunk
from app.graph.state import AgentState
from sqlmodel import select
from app.models.order import Order

SIMILARITY_THRESHOLD = 0.5  # 相似度阈值，低于此值则认为不相关

# ==========================================
# 组件初始化 (单例模式)
# ==========================================

# 1. Embedding 模型 (用于把用户问题转成向量)
embedding_model = OpenAIEmbeddings(
    base_url=settings.OPENAI_BASE_URL,
    api_key=SecretStr(settings.OPENAI_API_KEY),
    model=settings.EMBEDDING_MODEL,
    check_embedding_ctx_length=False
)

# 2. LLM 模型 (用于生成回答)
# temperature=0: 政策咨询需要严谨，不要发散
llm = ChatOpenAI(
    base_url=settings.OPENAI_BASE_URL,
    api_key=SecretStr(settings.OPENAI_API_KEY),
    model=settings.LLM_MODEL,
    temperature=0 
)

# 3. Prompt 模板
# 核心指令：严禁胡编乱造 (Hallucination Guardrails)
PROMPT_TEMPLATE = """
你是一个专业的电商政策咨询专家。请基于以下检索到的 context 回答用户的问题。

规则：
1. 只能依据 context 中的信息回答。
2. 如果 context 为空或没有相关信息，请直接回答“抱歉，暂未查询到相关规定”，严禁编造。
3. 语气专业、客气。

Context:
{context}

User Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# ==========================================
# 节点函数定义
# ==========================================

async def retrieve(state: AgentState) -> dict:
    """
    检索节点：带阈值过滤的硬逻辑
    """
    question = state["question"]
    print(f"🔍 [Retrieve] 正在检索: {question}")

    query_vector = await embedding_model.aembed_query(question)

    async with async_session_maker() as session:
        # 核心修改：我们在 select 时，把“距离数值”也查出来
        # logic: 查出对象 + 距离
        distance_col = KnowledgeChunk.embedding.cosine_distance(query_vector).label("distance") # type: ignore
        
        stmt = (
            select(KnowledgeChunk, distance_col) # 👈 多查一列
            .where(KnowledgeChunk.is_active) # type: ignore

            .order_by(distance_col)
            .limit(5) # 先多查几条，方便后面过滤
        )
        result = await session.exec(stmt)
        # 结果是 [(Chunk对象, 0.1), (Chunk对象, 0.2)...] 的元组列表
        results = result.all() 

    # --- 硬逻辑过滤 (Hard Filter) ---
    valid_chunks = []
    for chunk, distance in results:
        # 打印距离方便调试（生产环境可去掉）
        # distance 越小越好。如果 distance > 0.5，说明相关性很差
        print(f"   - 内容片段: {chunk.content[:10]}... | 距离分: {distance:.4f}")
        
        if distance < SIMILARITY_THRESHOLD:
            valid_chunks.append(chunk.content)
        else:
            print(f"   ❌ 距离过大，已丢弃")

    # 4. 更新 State
    # 如果 valid_chunks 为空，context 就是空列表 []
    # 此时 Prompt 里的 {context} 就会是空的，LLM 就会根据规则 2 回答“没找到”
    print(f"📄 [Retrieve] 最终有效记录: {len(valid_chunks)} 条")
    return {"context": valid_chunks}


GENERATE_PROMPT_TEMPLATE = """
你是一个电商客服助手。请根据提供的 [参考信息] 友好地回答用户。

[参考信息]：
{context_info}

[用户问题]：
{question}

规则：
1. 如果是订单信息，请清晰列出订单号、状态、总额和配送地址。
2. 如果是政策信息，请引用相关条款。
3. 如果参考信息为空，请礼貌地告知无法查到，并引导用户提供更多细节（如单号）。
4. 严禁编造数据库中不存在的订单状态。
"""

async def generate(state: AgentState) -> dict:
    print("🤖 [Generate] 正在生成综合回复...")
    
    # 1. 组装参考信息
    context_parts = []
    
    # 加入政策背景
    if state.get("context"):
        context_parts.append("【相关政策】:\n" + "\n".join(state["context"]))
    
    # 加入订单背景（更稳健的访问）
    if state.get("order_data"):
        order_raw = state["order_data"]
        # 如果传入的是 SQLModel 实例，尝试转为 dict
        if hasattr(order_raw, "model_dump"):
            order = order_raw.model_dump()
        else:
            order = order_raw or {}

        # 安全取字段（兼容多种可能的 key 名）
        def safe_get(d, *keys, default=None):
            if not isinstance(d, dict):
                return default
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        order_sn = safe_get(order, "order_sn", "sn", default="未知")
        status = safe_get(order, "status", default="未知")
        amount = safe_get(order, "total_amount", "amount", default=0)
        tracking = safe_get(order, "tracking_number", "tracking", "shipping_address", default=None)
        items = safe_get(order, "items", default=[])

        order_str = (
            f"【订单详情】:\n"
            f"- 订单号: {order_sn}\n"
            f"- 当前状态: {status}\n"
            f"- 订单金额: {amount} 元\n"
            f"- 收货地址: {tracking or '暂无'}\n"
            f"- 商品明细: {items}"
        )
        context_parts.append(order_str)

    context_info = "\n\n".join(context_parts) if context_parts else "暂无相关参考信息。"

    # 2. 构建回答
    # 注意：这里我们使用了 history，让 LLM 具备多轮感知
    messages = [
        SystemMessage(content=GENERATE_PROMPT_TEMPLATE.format(
            context_info=context_info,
            question=state["question"]
        ))
    ]
    
    # 可以在这里加入 state["history"] 的末尾几条，实现多轮对话
    
    response = await llm.ainvoke(messages)
    
    return {"answer": response.content}



# 意图识别的 System Prompt
INTENT_PROMPT = """你是一个电商客服分类器。你的任务是根据用户的输入，将其归类为以下三种意图之一：
- "ORDER": 用户询问关于他们自己的订单状态、物流、详情等。
- "POLICY": 用户询问关于平台通用的退换货、运费、时效等政策信息。
- "OTHER": 用户进行闲聊、打招呼或提出与上述无关的问题。

只返回分类标签（ORDER/POLICY/OTHER），不要返回任何其他文字。"""

async def intent_router(state: AgentState):
    """
    意图识别节点：判断用户想干什么
    """
    print(f"🧠 [Router] 正在分析意图: {state['question']}")
    
    response = await llm.ainvoke([
        SystemMessage(content=INTENT_PROMPT),
        HumanMessage(content=state["question"])
    ])
    
    intent = response.content.strip().upper()
    # 容错处理
    if intent not in ["ORDER", "POLICY", "OTHER"]:
        intent = "OTHER"
        
    print(f"🎯 [Router] 识别结果: {intent}")
    return {"intent": intent}

async def query_order(state: AgentState):
    """
    订单查询节点：不仅查询数据，还负责将数据“语言化”供生成节点使用
    """
    question = state["question"]
    user_id = state["user_id"]
    
    import re
    order_sn_match = re.search(r'SN\d+', question.upper())
    
    # 构造查询
    if not order_sn_match:
        print("🔎 [QueryOrder] 获取用户最近订单")
        stmt = (
            select(Order)
            .where(Order.user_id == user_id)
            .order_by(Order.created_at.desc())
            .limit(1)
        )
    else:
        order_sn = order_sn_match.group()
        print(f"🔎 [QueryOrder] 查询订单号: {order_sn}")
        stmt = select(Order).where(
            Order.order_sn == order_sn,
            Order.user_id == user_id 
        )

    async with async_session_maker() as session:
        # 修正点：使用 session.exec 更加简洁且符合 SQLModel 异步规范
        result = await session.exec(stmt)
        order = result.first()

    if not order:
        return {
            "order_data": None, 
            "context": ["用户询问了订单，但数据库中未查到相关记录。"]
        }
    
    # 这样 generate 节点就可以像处理 RAG 数据一样处理订单数据
    items_str = ", ".join([f"{i['name']}(x{i['qty']})" for i in order.items])
    order_context = (
        f"订单号: {order.order_sn}\n"
        f"状态: {order.status}\n"
        f"商品: {items_str}\n"
        f"金额: {order.total_amount}元\n"
        f"物流单号: {order.tracking_number or '暂无'}"
    )
    
    return {
        "order_data": order.model_dump(), 
        "context": [order_context] # 将订单信息放入 context，统一交给 generate 处理
    }

