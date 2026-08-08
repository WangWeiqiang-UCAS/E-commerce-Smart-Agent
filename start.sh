#!/bin/bash
# Local development bootstrap for the commerce-integrated Agent stack.
set -euo pipefail

echo " Starting E-commerce Smart Agent local stack..."
export PYTHONPATH=$PWD

poetry install

# Infrastructure: PostgreSQL + pgvector, Redis, RabbitMQ, MySQL, MongoDB.
echo " Starting infrastructure..."
docker compose up -d db redis rabbitmq mall-mysql mall-mongo

echo " Waiting for infrastructure to become healthy..."
sleep 8

echo " Applying Agent PostgreSQL migrations..."
poetry run alembic upgrade head

echo " Starting FastAPI Agent runtime on :8000..."
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

echo " Starting commerce event consumer..."
poetry run python scripts/consume_commerce_events.py &
CONSUMER_PID=$!

trap "kill $FASTAPI_PID $CONSUMER_PID; exit" INT
wait
