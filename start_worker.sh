echo " 启动 Celery Worker..."

celery -A app.celery_app worker \
  --loglevel=info \
  --concurrency=4 \
  --pool=prefork \
  -E