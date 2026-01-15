# app/api/v1/chat.py
import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from app.core.security import get_current_user_id
from app.api.v1.schemas import ChatRequest
# 直接从 workflow 模块导入 app_graph (它会在 main.py 启动时被赋值)
from app.graph.workflow import app_graph # 确保 app_graph 已被正确导入和初始化
from langchain_core.runnables import RunnableConfig 

router = APIRouter()

@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user_id: int = Depends(get_current_user_id)
):
    # 确保 app_graph 已经被编译 (这个检查仍然重要)
    if app_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not fully initialized. Please try again in a moment."
        )

    async def event_generator():
        thread_id = f"{current_user_id}_{request.thread_id}" 
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "question": request.question,
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
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text-event-stream")