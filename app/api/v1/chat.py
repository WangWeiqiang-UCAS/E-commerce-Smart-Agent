# app/api/v1/chat.py
import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from app.core.security import get_current_user_id
from app.api.v1.schemas import ChatRequest
from langchain_core.runnables import RunnableConfig 

router = APIRouter()

@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user_id: int = Depends(get_current_user_id)
):
    """
    聊天接口：支持订单查询和政策咨询
    
    - ORDER:  查询用户自己的订单
    - POLICY: 从知识库检索政策信息
    """
    # 在函数内部导入，避免模块加载顺序问题
    from app.graph.workflow import app_graph
    
    if app_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not fully initialized. Please try again in a moment."
        )

    async def event_generator():
        """SSE 流式响应生成器"""
        from app.graph.workflow import app_graph
        
        thread_id = f"{current_user_id}_{request.thread_id}" 
        config: RunnableConfig = {"configurable": {"thread_id":  thread_id}}

        initial_state = {
            "question":  request.question,
            "user_id": current_user_id,
            "history": [], 
            "context": [],
            "order_data": None,
            "answer": ""
        }

        try:
            async for event in app_graph.astream_events(
                initial_state, config, version="v2"
            ):
                kind = event["event"]
                
                # 只处理 LLM 流式输出
                if kind == "on_chat_model_stream":
                    data = event.get("data")
                    if data and isinstance(data, dict):
                        chunk = data.get("chunk")
                        if chunk:
                            content = chunk.content
                            if content:
                                payload = json.dumps({"token": content}, ensure_ascii=False)
                                yield f"data: {payload}\n\n"

            yield "data: [DONE]\n\n"
            
        except Exception as e: 
            error_msg = json.dumps({'error': str(e)}, ensure_ascii=False)
            yield f"data: {error_msg}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")