# app/api/v1/admin.py
"""
管理员 API
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from app.core.security import get_current_user_id  
from app.core.database import async_session_maker
from app.models.audit import AuditLog, AuditAction, RiskLevel
from app.models.refund import RefundApplication, RefundStatus
from app.models.message import MessageCard, MessageType, MessageStatus
from app.websocket.manager import manager
from app.tasks.refund_tasks import process_refund_payment, send_refund_sms
from sqlmodel import select, desc, or_

router = APIRouter()


class AuditTask(BaseModel):
    """审核任务"""
    audit_log_id: int
    thread_id: str
    user_id: int
    refund_application_id: Optional[int]
    order_id: Optional[int]
    trigger_reason: str
    risk_level: str
    context_snapshot: Dict[str, Any]
    created_at: str


class AdminDecisionRequest(BaseModel):
    """管理员决策请求"""
    action: str  # "APPROVE" | "REJECT"
    admin_comment: Optional[str] = None


class AdminDecisionResponse(BaseModel):
    """管理员决策响应"""
    success: bool
    message: str
    audit_log_id: int
    action: str


@router.get("/admin/tasks", response_model=List[AuditTask])
async def get_pending_tasks(
    risk_level: Optional[str] = None,
    current_admin_id: int = Depends(get_current_user_id)  # TODO: 改为管理员认证
):
    """
    获取待审核任务列表
    
    Query Params:
        risk_level: 可选，筛选风险等级 (HIGH, MEDIUM, LOW)
    """
    async with async_session_maker() as session:
        # 构建查询
        stmt = select(AuditLog).where(
            AuditLog.action == AuditAction.PENDING
        ).order_by(desc(AuditLog.created_at))
        
        if risk_level: 
            stmt = stmt.where(AuditLog.risk_level == risk_level)
        
        result = await session.execute(stmt)
        audit_logs = result.scalars().all()
        
        # 转换为响应格式
        tasks = []
        for log in audit_logs: 
            tasks.append(AuditTask(
                audit_log_id=log.id,
                thread_id=log.thread_id,
                user_id=log.user_id,
                refund_application_id=log.refund_application_id,
                order_id=log.order_id,
                trigger_reason=log.trigger_reason,
                risk_level=log.risk_level,
                context_snapshot=log.context_snapshot,
                created_at=log.created_at.isoformat(),
            ))
        
        return tasks


@router.post("/admin/resume/{audit_log_id}", response_model=AdminDecisionResponse)
async def admin_decision(
    audit_log_id: int,
    request: AdminDecisionRequest,
    current_admin_id: int = Depends(get_current_user_id)  # TODO: 改为管理员认证
):
    """
    管理员决策接口
    
    Path Params:
        audit_log_id: 审计日志ID
    
    Body:
        action:  APPROVE | REJECT
        admin_comment:  管理员备注
    """
    async with async_session_maker() as session:
        # 1. 查询审计日志
        result = await session.execute(
            select(AuditLog).where(AuditLog.id == audit_log_id)
        )
        audit_log = result.scalar_one_or_none()
        
        if not audit_log:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Audit log not found"
            )
        
        if audit_log.action != AuditAction.PENDING: 
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This audit has already been processed"
            )
        
        # 2. 更新审计日志
        action_enum = AuditAction.APPROVE if request.action == "APPROVE" else AuditAction.REJECT
        audit_log.action = action_enum
        audit_log.admin_id = current_admin_id
        audit_log.admin_comment = request.admin_comment
        audit_log.reviewed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        
        session.add(audit_log)
        
        # 3. 更新退款申请状态
        if audit_log.refund_application_id:
            refund_result = await session.execute(
                select(RefundApplication).where(
                    RefundApplication.id == audit_log.refund_application_id
                )
            )
            refund = refund_result.scalar_one_or_none()
            
            if refund:
                if action_enum == AuditAction.APPROVE:
                    refund.status = RefundStatus.APPROVED
                    refund.admin_note = request.admin_comment
                    refund.reviewed_by = current_admin_id
                    refund.reviewed_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    
                    # 4. 触发异步任务：退款 + 短信通知
                    process_refund_payment.delay(
                        refund_id=refund.id,
                        amount=float(refund.refund_amount),
                        payment_method="原支付方式"
                    )
                    
                    send_refund_sms.delay(
                        refund_id=refund.id,
                        phone="138****1234",  # TODO: 从用户表获取
                        message=f"您的退款申请已通过，退款金额¥{refund.refund_amount}将在3-5个工作日退回。"
                    )
                    
                else: 
                    refund.status = RefundStatus.REJECTED
                    refund.admin_note = request.admin_comment
                    refund.reviewed_by = current_admin_id
                    refund.reviewed_at = datetime.now(timezone.utc).replace(tzinfo=None)
                
                session.add(refund)
        
        # 5. 创建状态变更消息卡片
        status_message = " 审核通过，资金将在3-5个工作日内原路退回" if action_enum == AuditAction.APPROVE else f" 审核未通过: {request.admin_comment}"
        
        message_card = MessageCard(
            thread_id=audit_log.thread_id,
            message_type=MessageType.AUDIT_CARD,
            status=MessageStatus.SENT,
            content={
                "card_type": "audit_result",
                "action": request.action,
                "message": status_message,
                "admin_comment": request.admin_comment,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            sender_type="admin",
            sender_id=current_admin_id,
            receiver_id=audit_log.user_id,
        )
        session.add(message_card)
        
        await session.commit()
        
        # 6. 通过 WebSocket 实时推送状态变更
        await manager.notify_status_change(
            thread_id=audit_log.thread_id,
            status=request.action,
            data={
                "message": status_message,
                "admin_comment": request.admin_comment,
            }
        )
        
        return AdminDecisionResponse(
            success=True,
            message=f"审核决策已提交:  {request.action}",
            audit_log_id=audit_log_id,
            action=request.action,
        )