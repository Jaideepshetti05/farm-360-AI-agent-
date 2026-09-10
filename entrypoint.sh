#!/bin/sh
set -e

echo "==> [Farm360 Boot] Running database migrations (alembic upgrade head)..."
python -m alembic upgrade head

echo "==> [Farm360 Boot] Database migrations applied successfully."
echo "==> [Farm360 Boot] Starting FastAPI application with Uvicorn..."
exec uvicorn backend.app:app --host 0.0.0.0 --port 8000 "$@"
