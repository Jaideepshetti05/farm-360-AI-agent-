#!/bin/sh
set -e

echo "==> [Farm360 Boot] Starting backend entrypoint..."

# ── 1. Wait for PostgreSQL readiness ──────────────────────────────────────────
python - << 'EOF'
import os
import sys
import time
import asyncio
from urllib.parse import urlparse

raw_url = os.environ.get("DATABASE_URL", "").strip()
if not raw_url or raw_url.startswith("sqlite"):
    print("==> [Farm360 Boot] SQLite / local DB detected. Skipping PostgreSQL readiness check.")
    sys.exit(0)

clean_url = raw_url.replace("+asyncpg", "").replace("+psycopg2", "").replace("+aiosqlite", "")
parsed = urlparse(clean_url)

host = parsed.hostname or "postgres"
port = parsed.port or 5432
database = parsed.path.lstrip("/") or "farm360"
username = parsed.username or "postgres"

sanitized_target = f"{username}@{host}:{port}/{database}"
max_retries = int(os.environ.get("DB_MAX_RETRIES", 30))
retry_interval = float(os.environ.get("DB_RETRY_INTERVAL", 2.0))

print(f"==> [Farm360 Boot] Waiting for PostgreSQL readiness at {sanitized_target} (max {max_retries} attempts, {retry_interval}s interval)...")

try:
    import asyncpg
except ImportError:
    print("==> [Farm360 Boot] WARNING: asyncpg not installed, falling back to socket connectivity check.")
    import socket
    for attempt in range(1, max_retries + 1):
        try:
            with socket.create_connection((host, int(port)), timeout=2.0):
                print(f"==> [Farm360 Boot] PostgreSQL port {port} is reachable (socket check).")
                sys.exit(0)
        except (socket.error, OSError) as err:
            print(f"==> [Farm360 Boot] PostgreSQL not ready (attempt {attempt}/{max_retries}): {err}")
            time.sleep(retry_interval)
    print(f"==> [Farm360 Boot] ERROR: PostgreSQL at {sanitized_target} unreachable after {max_retries} attempts.")
    sys.exit(1)

async def check_async():
    for attempt in range(1, max_retries + 1):
        try:
            conn = await asyncpg.connect(clean_url, timeout=3.0)
            await conn.close()
            print(f"==> [Farm360 Boot] PostgreSQL is online and accepting connections ({sanitized_target}).")
            return 0
        except Exception as e:
            raw_err = str(e).split("\n")[0]
            err_name = type(e).__name__
            err_msg = f"{err_name}: {raw_err}" if raw_err else err_name
            if parsed.password:
                err_msg = err_msg.replace(parsed.password, "******")
            print(f"==> [Farm360 Boot] PostgreSQL not ready (attempt {attempt}/{max_retries}): {err_msg}")
            await asyncio.sleep(retry_interval)
    print(f"==> [Farm360 Boot] ERROR: PostgreSQL at {sanitized_target} failed to become ready after {max_retries} attempts.")
    return 1

sys.exit(asyncio.run(check_async()))
EOF

echo "==> [Farm360 Boot] Database is ready. Running migrations (alembic upgrade head)..."
python -m alembic upgrade head

echo "==> [Farm360 Boot] Database migrations applied successfully."
echo "==> [Farm360 Boot] Starting FastAPI application with Uvicorn..."
exec uvicorn backend.app:app --host 0.0.0.0 --port 8000 "$@"

