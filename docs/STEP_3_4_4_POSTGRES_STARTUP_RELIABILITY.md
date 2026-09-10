# Step 3.4.4 — PostgreSQL Startup Reliability

## 1. Objective
Ensure the Farm360 backend container is resilient to PostgreSQL startup timing, container ordering, and network race conditions in local Docker Compose, staging environments, and production AWS EC2 deployments. The backend container must explicitly wait for PostgreSQL to accept connections before invoking Alembic database migrations (`python -m alembic upgrade head`) and launching Uvicorn.

---

## 2. Existing Startup Problem
Prior to this step, `entrypoint.sh` immediately executed:
```sh
python -m alembic upgrade head
```
at container launch. 

In Docker Compose and multi-container environments, container dependencies (`depends_on: - postgres`) only ensure that the PostgreSQL container process has started, not that the PostgreSQL daemon inside has completed initialization, recovery, or SSL/socket binding. If Alembic attempts to connect while PostgreSQL is in the initialization or startup phase:
- Alembic crashes with a connection failure (`ConnectionRefusedError`, `CannotConnectNowError`, or `ConnectionResetError`).
- Under `set -e`, the entire backend container aborts immediately and enters a `CrashLoopBackOff` state.
- In automated deployments (e.g. AWS EC2, ECS), premature backend crashes cause health check timeouts and deployment rollbacks.

---

## 3. Readiness Strategy
Instead of introducing external OS dependencies (such as `netcat`, `curl`, or `postgresql-client`), the readiness check utilizes the existing, pre-installed Python runtime and `asyncpg` driver:

1. **Protocol & Environment Agnostic:**
   - Detects `DATABASE_URL` from the environment.
   - If `DATABASE_URL` is unset or points to SQLite (`sqlite://` or `sqlite+aiosqlite://`), the readiness check skips immediately with code `0`.
   - Strips dialect specifiers (`+asyncpg`, `+psycopg2`) to produce standard DSNs compatible with `asyncpg.connect()`.
2. **True Application-Level Healthcheck:**
   - Raw TCP socket checks can pass prematurely while PostgreSQL is still initializing ("the database system is starting up").
   - `asyncpg.connect()` executes a full handshake and protocol handshake against PostgreSQL, ensuring the database is genuinely ready to process queries.
3. **Graceful Fallback:**
   - If `asyncpg` is unavailable in the environment, the script gracefully falls back to non-blocking TCP socket polling (`socket.create_connection`).
4. **Deterministic Fail-Fast Pipeline:**
   ```
   Backend Container Starts
             ↓
   Parse DATABASE_URL & Config
             ↓
   PostgreSQL Readiness Loop (asyncpg, max retries)
             ↓
   Run Alembic Migrations (`alembic upgrade head`)
             ↓
   Start FastAPI Server (`exec uvicorn ...`)
   ```

---

## 4. Retry Configuration
The readiness loop parameters are configurable via environment variables, with sensible production defaults:

| Variable | Default Value | Description |
|---|---|---|
| `DB_MAX_RETRIES` | `30` | Maximum number of connection attempts before failing |
| `DB_RETRY_INTERVAL` | `2.0` | Delay in seconds between consecutive connection attempts |
| `Connection Timeout` | `3.0s` | Per-attempt `asyncpg.connect(..., timeout=3.0)` threshold |

With default settings:
- Total wait budget: `30 attempts * 2.0s = 60.0 seconds` (plus connection timeouts).
- Ample buffer for PostgreSQL initial volume initialization, extension loading (e.g., `pgvector`), and recovery without waiting indefinitely.

---

## 5. Security Considerations
- **No Password Leakage:** Credentials in `DATABASE_URL` are strictly sanitized. The target host log outputs only `username@host:port/database`.
- **Exception Sanitization:** Connection exceptions (e.g., from `asyncpg` or OS sockets) are scrubbed so that if any connection string or password appears in the error string, it is redacted with `******`.
- **No Insecure Defaults:** The script does not hardcode passwords, hostnames, or credentials. It reads solely from the injected `DATABASE_URL`.
- **No Secret Environment Dumping:** No debugging statements print raw environment maps or full URIs.

---

## 6. Changes Made

### File Modified: [`entrypoint.sh`](file:///c:/Users/Jaideep/Desktop/ml%20models/entrypoint.sh)
- Added inline Python readiness check prior to running Alembic migrations.
- Retained strict shell behavior (`set -e`) so that any readiness check failure or migration failure prevents Uvicorn from starting.
- Preserved existing Uvicorn invocation (`exec uvicorn backend.app:app --host 0.0.0.0 --port 8000 "$@"`).
- Preserved Unix line endings (`LF`).

---

## 7. Verification
Static and unit verification tests were executed against the Python readiness logic using an automated test harness (`verify_entrypoint.py`):

1. **SQLite Bypass Test:**
   - Input: `DATABASE_URL=sqlite+aiosqlite:///./test.db`
   - Output: `==> [Farm360 Boot] SQLite / local DB detected. Skipping PostgreSQL readiness check.`
   - Exit Code: `0` (Success)
2. **Unreachable PostgreSQL Test with Password Scrubbing:**
   - Input: `DATABASE_URL=postgresql+asyncpg://admin_user:super_secret_password_xyz@127.0.0.1:59999/farm360_db`, `DB_MAX_RETRIES=2`, `DB_RETRY_INTERVAL=0.5`
   - Output:
     ```
     ==> [Farm360 Boot] Waiting for PostgreSQL readiness at admin_user@127.0.0.1:59999/farm360_db (max 2 attempts, 0.5s interval)...
     ==> [Farm360 Boot] PostgreSQL not ready (attempt 1/2): ConnectionRefusedError: ...
     ==> [Farm360 Boot] PostgreSQL not ready (attempt 2/2): ConnectionRefusedError: ...
     ==> [Farm360 Boot] ERROR: PostgreSQL at admin_user@127.0.0.1:59999/farm360_db failed to become ready after 2 attempts.
     ```
   - Exit Code: `1` (Failure)
   - Secret Check: Verified `super_secret_password_xyz` was not present in stdout/stderr.
3. **Syntax & Line Endings:**
   - Confirmed 0 CRLF characters, 76 LF characters.
   - POSIX compliant syntax for `/bin/sh`.

---

## 8. Docker Runtime Results
- **Docker Daemon Status:** Docker Desktop daemon was not running on the local Windows host during verification (`failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`).
- **Runtime Verification Status:** Docker runtime container execution test is **PENDING** until Docker Desktop is started or during live deployment on the AWS EC2 staging host.
- **Static & Process Verification:** Complete and fully passing.

---

## 9. Failure Behavior
If PostgreSQL fails to become ready within the allotted retry budget (`DB_MAX_RETRIES`):
1. Script prints: `==> [Farm360 Boot] ERROR: PostgreSQL at <sanitized_target> failed to become ready after <N> attempts.`
2. Python sub-process exits with code `1`.
3. Due to `set -e` in `entrypoint.sh`, the shell terminates immediately.
4. Alembic migrations (`python -m alembic upgrade head`) are **never executed** against an unreachable database.
5. Uvicorn is **never started** in a broken state.
6. Container exits with non-zero exit code, triggering appropriate orchestrator restart/alert policies.

---

## 10. Remaining Limitations
- While `entrypoint.sh` guarantees connection readiness before migrations, it relies on environment-injected `DATABASE_URL`. If invalid connection credentials are provided, it will exhaust retries and fail fast.
- Docker daemon must be running to execute live multi-container integration tests.

---

## 11. Next Deployment Task
**Step 3.4.5 — Production Compose + Caddy**
- Implement `docker-compose.prod.yml` with proper service dependencies, environment variable mapping, volume persistence, and Caddy reverse proxy for automated HTTPS.
