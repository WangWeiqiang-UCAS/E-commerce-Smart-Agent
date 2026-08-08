# app/api/v1/websocket.py
"""
WebSocket 路由
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.core.security import get_current_user_id_ws, get_current_admin_id_ws
from app.websocket.manager import manager

router = APIRouter()


@router.websocket("/ws/{thread_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    thread_id: str,
    token: str = Query(...),
):
    """
    用户 WebSocket 连接
    
    Query Params:
        token: JWT Token
    """
    try:
        # 验证 Token
        user_id = await get_current_user_id_ws(token)
        
        # 建立连接
        await manager.connect_user(websocket, user_id, thread_id)
        
        try:
            while True:
                # 接收心跳或其他消息
                data = await websocket.receive_text()
                
                # 处理心跳
                if data == "ping":
                    await websocket.send_text("pong")
                
        except WebSocketDisconnect: 
            manager.disconnect_user(user_id, thread_id)
            
    except Exception as e:
        print(f" [WS] 连接错误: {e}")
        await websocket.close(code=1008, reason=str(e))


@router.websocket("/ws/admin/{admin_id}")
async def admin_websocket_endpoint(
    websocket: WebSocket,
    admin_id: int,
    token: str = Query(...),
):
    """
    管理员 WebSocket 连接
    
    Query Params: 
        token: JWT Token (需验证管理员权限)
    """
    try:
        authenticated_admin_id = await get_current_admin_id_ws(token)
        if authenticated_admin_id != admin_id:
            raise RuntimeError("Administrator identity does not match the requested channel")
        
        # 建立连接
        await manager.connect_admin(websocket, admin_id)
        
        try:
            while True: 
                data = await websocket.receive_text()
                
                if data == "ping":
                    await websocket.send_text("pong")
                
        except WebSocketDisconnect:
            manager.disconnect_admin(admin_id)
            
    except Exception as e:
        print(f" [WS] 管理员连接错误: {e}")
        await websocket.close(code=1008, reason=str(e))
