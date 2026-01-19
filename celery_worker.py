import sys
import os
from app.celery_app import celery_app

# 简单判断系统
is_windows = sys.platform.startswith('win')

# 根据系统选择执行池
pool_type = 'solo' if is_windows else 'prefork'

if __name__ == "__main__":
    celery_app.worker_main([
        "worker",
        "--loglevel=info",
        "--concurrency=4",
        f"--pool={pool_type}", # 动态设置
        "-E",
    ])