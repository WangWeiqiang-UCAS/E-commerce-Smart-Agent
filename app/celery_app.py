# app/celery_app.py
"""
Celery 异步任务系统
处理短信发送、退款网关调用等耗时操作
"""
from celery import Celery
from app.core.config import settings

# 创建 Celery 实例
celery_app = Celery(
    "ecommerce_agent",
    broker=settings.CELERY_BROKER,
    backend=settings.CELERY_BACKEND,
)

# Celery 配置
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5分钟超时
    task_soft_time_limit=240,  # 4分钟软超时
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
)

# 自动发现任务
celery_app.autodiscover_tasks(["app.tasks"])