#!/bin/bash
# 启动所有 v4.0 服务

echo " 启动 E-commerce Smart Agent v4.0 所有服务..."
export PYTHONPATH=$PWD

# 检查依赖
echo " 检查依赖..."
poetry install

# 启动 Redis 和 PostgreSQL
echo " 启动基础设施..."
docker-compose up -d postgres redis

# 等待数据库就绪
echo " 等待数据库启动..."
sleep 5

# 执行数据库迁移
echo " 执行数据库迁移..."
poetry run alembic upgrade head

# 启动服务（使用 tmux 或单独终端）
echo " 启动 FastAPI 服务..."
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$! 

echo " 启动 Celery Worker..."
poetry run celery -A app.celery_app worker --loglevel=info --concurrency=4 --pool=solo &
CELERY_PID=$!

sleep 3

echo " 启动用户界面..."
poetry run python app/frontend/customer_ui.py &
UI_PID=$!

echo " 启动管理员工作台..."
poetry run python app/frontend/admin_dashboard.py &
ADMIN_PID=$!

echo ""
echo " 所有服务已启动！"
echo ""
echo " 访问地址:"
echo "  - FastAPI API: http://localhost:8000"
echo "  - API 文档: http://localhost:8000/docs"
echo "  - 用户界面: http://localhost:7860"
echo "  - 管理员工作台: http://localhost:7861"
echo ""
echo " 按 Ctrl+C 停止所有服务"

# 等待中断信号
trap "kill $FASTAPI_PID $CELERY_PID $UI_PID $ADMIN_PID; exit" INT
wait