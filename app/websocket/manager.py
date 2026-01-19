# app/websocket/manager.py
"""
WebSocket 连接管理器
负责维护客户端连接、广播消息、状态同步
"""
from typing import Dict, Set, Optional
from fastapi import WebSocket
import json
import asyncio
from datetime import datetime


class ConnectionManager:
    """WebSocket 连接管理器"""
    
    def __init__(self):
        # 用户连接池:  {user_id: {thread_id: WebSocket}}
        self.active_connections: Dict[int, Dict[str, WebSocket]] = {}
        
        # 管理员连接池: {admin_id: WebSocket}
        self.admin_connections: Dict[int, WebSocket] = {}
        
        # 线程订阅: {thread_id:  Set[WebSocket]}
        self.thread_subscribers: Dict[str, Set[WebSocket]] = {}
    
    async def connect_user(self, websocket: WebSocket, user_id: int, thread_id: str):
        """用户连接"""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = {}
        
        self.active_connections[user_id][thread_id] = websocket
        
        # 订阅线程
        if thread_id not in self.thread_subscribers:
            self.thread_subscribers[thread_id] = set()
        self.thread_subscribers[thread_id].add(websocket)
        
        print(f" [WS] 用户 {user_id} 连接到会话 {thread_id}")
    
    async def connect_admin(self, websocket:  WebSocket, admin_id: int):
        """管理员连接"""
        await websocket.accept()
        self.admin_connections[admin_id] = websocket
        print(f" [WS] 管理员 {admin_id} 已连接")
    
    def disconnect_user(self, user_id: int, thread_id:  str):
        """断开用户连接"""
        if user_id in self.active_connections:
            ws = self.active_connections[user_id].pop(thread_id, None)
            if ws and thread_id in self.thread_subscribers:
                self.thread_subscribers[thread_id].discard(ws)
            
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        print(f" [WS] 用户 {user_id} 断开会话 {thread_id}")
    
    def disconnect_admin(self, admin_id: int):
        """断开管理员连接"""
        self.admin_connections.pop(admin_id, None)
        print(f" [WS] 管理员 {admin_id} 已断开")
    
    async def send_to_user(self, user_id: int, thread_id: str, message: dict):
        """发送消息给指定用户"""
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id].get(thread_id)
            if websocket: 
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    print(f" [WS] 发送失败: {e}")
                    self.disconnect_user(user_id, thread_id)
    
    async def send_to_thread(self, thread_id: str, message: dict):
        """广播消息到指定会话的所有订阅者"""
        if thread_id in self.thread_subscribers:
            disconnected = set()
            for websocket in self.thread_subscribers[thread_id]:
                try: 
                    await websocket.send_json(message)
                except Exception as e:
                    print(f" [WS] 广播失败:  {e}")
                    disconnected.add(websocket)
            
            # 清理断开的连接
            self.thread_subscribers[thread_id] -= disconnected
    
    async def broadcast_to_admins(self, message: dict):
        """广播消息给所有管理员"""
        disconnected = []
        for admin_id, websocket in self.admin_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f" [WS] 管理员 {admin_id} 发送失败: {e}")
                disconnected.append(admin_id)
        
        # 清理断开的连接
        for admin_id in disconnected:
            self.disconnect_admin(admin_id)
    
    async def notify_status_change(self, thread_id: str, status: str, data: Optional[dict] = None):
        """
        通知状态变更
        
        Args:
            thread_id: 会话ID
            status: 新状态 (e.g., "WAITING_ADMIN", "APPROVED", "REJECTED")
            data: 附加数据
        """
        message = {
            "type": "status_change",
            "thread_id": thread_id,
            "status": status,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # 通知该会话的所有订阅者
        await self.send_to_thread(thread_id, message)
        
        # 如果是需要审核的状态，同时通知管理员
        if status == "WAITING_ADMIN":
            await self.broadcast_to_admins({
                "type": "new_audit_task",
                "thread_id":  thread_id,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
            })


# 全局单例
manager = ConnectionManager()