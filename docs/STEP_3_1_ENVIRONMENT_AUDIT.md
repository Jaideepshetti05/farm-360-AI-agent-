# Farm360 DevOps — Step 3.1 Environment Architecture Audit

**Milestone:** Step 3 — Prepare Environments (DEV → STAGING → PROD)  
**Task:** Step 3.1 — Environment Architecture Audit  
**Mode:** READ-ONLY AUDIT  
**Date:** 2026-09-03  

---

## Executive Summary

Step 2.8 established a functional 4-container production baseline (`farm360-backend`, `farm360-frontend`, `postgres`, `redis`) with zero application regressions, validated model loading, verified live inference, and automated database migrations.

However, the codebase currently possesses **tightly coupled environment logic**, a **binary environment model** (`production` vs `development` with no first-class `staging`), **hardcoded assumptions regarding `.env` paths**, and **missing production features for cloud caching** (e.g., Redis password/TLS).

This audit documents the current state, identifies architecture gaps across DEV, STAGING, and PROD, and provides a safe, phased migration roadmap.

---

## A. Current Environment Model

Currently, Farm360 evaluates runtime environments through an ad-hoc and fragmented set of mechanisms:

### 1. Environment Variable Detection
* Evaluated primarily in [`backend/core/security.py`](file:///c:/Users/Jaideep/Desktop/ml%20models/backend/core/security.py) via `is_production()`:
  ```python
  def is_production() -> bool:
      env = (
          os.environ.get("ENVIRONMENT")
          or os.environ.get("APP_ENV")
          or os.environ.get("ENV")
          or ""
      ).strip().lower()
      if env in ("production", "prod", "live"):
          return True
      # Also treat PostgreSQL deployment as production
      db_url = os.environ.get("DATABASE_URL", "").strip().lower()
      if db_url.startswith("postgresql") or db_url.startswith("postgres"):
          return True
      return False
  ```
* **Critical Finding:** `is_production()` forces production mode whenever `DATABASE_URL` starts with `postgresql` or `postgres`, regardless of whether `ENVIRONMENT=development` or `ENVIRONMENT=staging`. This creates a false production state during local Docker development.
* **No First-Class Staging Concept:** The code only supports a binary condition: `is_production()` is either `True` or `False`. `staging` is not recognized as a distinct environment with intermediate policies (e.g. non-local CORS combined with sanitized staging data).

### 2. Configuration Injection & Precedence
* [`backend/config.py`](file:///c:/Users/Jaideep/Desktop/ml%20models/backend/config.py) uses Pydantic's `BaseSettings`, but explicitly calls:
  ```python
  root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
  load_dotenv(os.path.join(root_dir, ".env"))
  ```
  It has no built-in mechanism to load `.env.development`, `.env.staging`, or `.env.production` based on an active profile.
* `Settings` in `backend/config.py` does not define `environment`, `database_url`, `redis_host`, or `cors_origins`. Those are read ad-hoc via `os.environ.get(...)` across different modules.

### 3. Fail-Fast Policies
* **In Production (`is_production() == True`):**
  - `CORS_ORIGINS` must be explicitly defined and cannot contain wildcards (`*`).
  - `FARM360_ENCRYPTION_KEY` must be a valid 32-byte Fernet key; otherwise, startup halts immediately.
* **In Development (`is_production() == False`):**
  - `CORS_ORIGINS` defaults to `["http://localhost:3000", "http://127.0.0.1:3000"]` if unset.
  - `FARM360_ENCRYPTION_KEY` falls back to an in-memory XOR key with a warning.
  - `DATABASE_URL` falls back to SQLite (`sqlite+aiosqlite:///./farm360.db`) if unset.

---

## B. Configuration Matrix

| Configuration Variable | DEV (Local / Docker) | STAGING (Pre-Prod) | PROD (Production) | Secret? | Current Source / Resolution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ENVIRONMENT`** | `development` | `staging` | `production` | No | `docker-compose.yml`, `.env`, `security.py` |
| **`FARM360_API_KEY`** | `dev-secret-key` | Dedicated staging secret | Cryptographically generated 32-byte secret | **YES** | `backend/config.py`, root `.env`, Compose |
| **`FARM360_ENCRYPTION_KEY`** | Optional (fallback XOR) | Dedicated 32-byte Fernet key | Dedicated KMS-managed 32-byte Fernet key | **YES** | `backend/core/security.py`, `.env` |
| **`DATABASE_URL`** | `sqlite+aiosqlite:///./farm360.db` (or local PostgreSQL) | `postgresql+asyncpg://.../farm360_staging` | `postgresql+asyncpg://.../farm360_prod` | **YES** | `backend/core/database.py`, Compose |
| **`REDIS_HOST`** | `127.0.0.1` or `redis` | Staging Redis host / cluster | Production Redis host / cluster | No | `backend/services/cache_service.py` |
| **`REDIS_PORT`** | `6379` | `6379` (or cloud port) | `6379` (or cloud port) | No | `backend/services/cache_service.py` |
| **`REDIS_DB`** | `0` | `0` | `0` | No | `backend/services/cache_service.py` |
| **`REDIS_PASSWORD`** | *(Not supported in code)* | Required in cloud | Required in cloud | **YES** | **MISSING IN CODE** |
| **`REDIS_SSL`** | `false` | `true` (managed redis) | `true` (managed redis) | No | **MISSING IN CODE** |
| **`CORS_ORIGINS`** | `http://localhost:3000,http://127.0.0.1:3000` | `https://staging.farm360.ai` | `https://app.farm360.ai,https://farm360.ai` | No | `backend/app.py` (`_resolve_cors_origins`) |
| **`BACKEND_API_URL`** | `http://127.0.0.1:8000` or `http://farm360-backend:8000` | `http://staging-backend:8000` or internal VPC DNS | `http://backend:8000` or internal VPC DNS | No | `frontend/src/app/api/.../route.ts` |
| **`NEXT_PUBLIC_API_BASE_URL`** | `http://127.0.0.1:8000` | Optional / Internal fallback | Optional / Internal fallback | No | `docker-compose.yml`, `frontend/.env.local` |
| **`GOOGLE_API_KEY_1..5`** | Developer test keys | Staging quota-controlled keys | Production monitored enterprise keys | **YES** | `backend/provider_manager.py`, `.env` |
| **`OPENROUTER_API_KEY_1..5`** | Developer test keys | Staging quota-controlled keys | Production monitored enterprise keys | **YES** | `backend/provider_manager.py`, `.env` |
| **`OPENAI_API_KEY_1..3`** | Developer test keys | Staging quota-controlled keys | Production monitored enterprise keys | **YES** | `backend/provider_manager.py`, `.env` |
| **`OPENWEATHER_API_KEY`** | Developer test key | Staging test key | Production API key | **YES** | `backend/config.py`, `.env` |
| **`MODEL_BASE_PATH`** | Repository root (or `/app`) | `/app` (container storage) | `/app` (container storage) | No | `backend/config.py`, `docker-compose.yml` |

---

## C. Current Problems & Vulnerabilities

### 🔴 Critical Issues

1. **`is_production()` Database Coupling:**
   In [`backend/core/security.py`](file:///c:/Users/Jaideep/Desktop/ml%20models/backend/core/security.py), line 30 returns `True` if `DATABASE_URL` contains `postgres`. Consequently:
   - Running PostgreSQL locally via Docker automatically forces production rules (`CORS_ORIGINS` and `FARM360_ENCRYPTION_KEY` strict enforcement).
   - Setting `ENVIRONMENT=staging` with PostgreSQL triggers production mode instead of staging policies.
2. **Missing Redis Authentication & TLS Support:**
   In [`backend/services/cache_service.py`](file:///c:/Users/Jaideep/Desktop/ml%20models/backend/services/cache_service.py) and [`backend/services/health_service.py`](file:///c:/Users/Jaideep/Desktop/ml%20models/backend/services/health_service.py), `redis.Redis()` only accepts `host`, `port`, and `db`. There is no support for `REDIS_PASSWORD`, `REDIS_USER`, or SSL (`rediss://`). Cloud providers (AWS ElastiCache, GCP Memorystore, Redis Cloud) require authentication and SSL, preventing deployment to real staging/production clusters.
3. **Absence of a Formal Staging Environment:**
   There is no explicit `staging` recognition. Staging requires production-grade security (strict origins, encryption) but needs separate databases, relaxed rate limits for testing, and mock/staging API keys.

---

### 🟠 High Severity Issues

4. **Docker Compose Mixes Development & Production Patterns:**
   [`docker-compose.yml`](file:///c:/Users/Jaideep/Desktop/ml%20models/docker-compose.yml) binds source code directories directly (`./frontend:/app`), builds Next.js on every container startup (`command: sh -c "npm install && npm run build && npm start"`), and exposes database port 5432 directly. Production requires immutable container images without host source binds.
5. **Hardcoded `.env` Path Loading:**
   `backend/config.py` hardcodes `load_dotenv(os.path.join(root_dir, ".env"))`. It does not support environment-specific files like `.env.development`, `.env.staging`, or `.env.production`.
6. **Frontend Base URL Inconsistency:**
   In `docker-compose.yml`, `NEXT_PUBLIC_API_BASE_URL` is set to `http://farm360-backend:8000`. While the current Next.js implementation proxies all calls through server-side route handlers, setting `NEXT_PUBLIC_*` to an internal Docker hostname is dangerous because client-side JavaScript executing in a user's browser cannot resolve `farm360-backend`.

---

### 🟡 Medium Severity Issues

7. **Default Host Discrepancy in Redis Clients:**
   `health_service.py` defaults to `REDIS_HOST="localhost"`, while `cache_service.py` defaults to `REDIS_HOST="127.0.0.1"`.
8. **Database Name Hardcoding Across Environments:**
   Compose and documentation assume database `farm360`. Staging and Production require distinct database names (e.g., `farm360_staging`, `farm360_prod`) to prevent accidental crossover when sharing a database server.
9. **Lack of Automated DB Seed Strategy:**
   While Alembic runs migrations automatically on container start, there is no structured seeding mechanism for development/staging initial state (e.g. test farmers, sample crops).

---

### 🟢 Low Severity Issues

10. **Obsolete Version Directive in Compose:**
    `version: '3.8'` generates deprecation warnings in current Docker Compose versions.
11. **No CI/CD Workflows:**
    There is no `.github/workflows/` directory to run automated linting, test suites, or multi-environment container image builds.

---

## D. Recommended Target Architecture

```
                 ┌────────────────────────────────────────────────────────┐
                 │                   Farm360 Environments                 │
                 └───────────────────────────┬────────────────────────────┘
                                             │
         ┌───────────────────────────────────┼───────────────────────────────────┐
         ▼                                   ▼                                   ▼
┌──────────────────┐               ┌──────────────────┐               ┌──────────────────┐
│   DEVELOPMENT    │               │     STAGING      │               │    PRODUCTION    │
├──────────────────┤               ├──────────────────┤               ├──────────────────┤
│ Local / Compose  │               │ Cloud Replica    │               │ Production Cloud │
│ SQLite / PG 16   │               │ Managed PG 16    │               │ HA PG 16 + Vector│
│ Local Redis      │               │ Managed Redis    │               │ Redis Cluster    │
│ Hot-Reload Code  │               │ Immutable Images │               │ Immutable Images │
│ Localhost CORS   │               │ Staging Domain   │               │ Production Domain│
│ Dev / Mock Keys  │               │ Staging Key Pool │               │ Prod Key Pool    │
└──────────────────┘               └──────────────────┘               └──────────────────┘
```

### 1. Environment Policy Matrix

| Feature / Policy | Development (DEV) | Staging (STAGING) | Production (PROD) |
| :--- | :--- | :--- | :--- |
| **`ENVIRONMENT` string** | `development` | `staging` | `production` |
| **Code Execution** | Hot-reloading / mounted volumes | Pre-built container images | Pre-built container images |
| **Database** | SQLite or local Docker PostgreSQL | Dedicated Staging PostgreSQL | Managed Cloud PostgreSQL (HA, automated backups) |
| **Database Migrations** | Tested with Alembic | Automated pre-deploy migration | Automated blue/green migration with rollback plan |
| **Redis** | Local container (no auth) | Password-protected Redis | Managed Redis with Password + TLS |
| **CORS Origins** | `localhost:3000`, `127.0.0.1:3000` | `staging.farm360.ai` | `app.farm360.ai`, `farm360.ai` |
| **Encryption Key** | Ephemeral fallback permitted | Mandatory Fernet 32-byte key | Mandatory Secret Manager / Fernet key |
| **Farm360 API Key** | Dev static key | Staging secret key | Production cryptographically secure key |
| **External LLM Keys** | Developer free/sandbox keys | Team shared staging keys | Enterprise quota production keys |
| **Telemetry / Logging** | `DEBUG` / `INFO` (stdout) | `INFO` (structured JSON) | `WARNING` / `ERROR` (structured JSON + APM) |

---

## E. Docker Compose Multi-Environment Strategy

### Recommendation: Base Compose + Environment Overlays

Rather than a single monolithic file or completely disjoint configurations, standard DevOps best practice is a **Base Compose file with environment overrides**:

1. **`docker-compose.yml` (Base):**
   - Defines standard service declarations: `postgres`, `redis`, `farm360-backend`, `farm360-frontend`.
   - Declares common health checks, restart policies, internal networks, and environment variable bindings.
2. **`docker-compose.override.yml` (Development Default):**
   - Automatically loaded by `docker compose up` on local developer machines.
   - Binds source code volumes (`./backend:/app/backend`, `./frontend:/app`).
   - Exposes local ports to host (`8000:8000`, `3000:3000`, `5432:5432`, `6379:6379`).
   - Enables development logs.
3. **`docker-compose.staging.yml` (Staging Overlay):**
   - Strips code mounts; runs immutable built images.
   - Binds to staging network / managed databases.
   - Restricts port exposure (e.g. only frontend port 3000 or behind reverse proxy).
4. **`docker-compose.prod.yml` (Production Overlay):**
   - Strips code mounts; runs version-tagged immutable images.
   - Externalizes PostgreSQL and Redis (relies on managed cloud services instead of local containers, if desired).
   - Attaches production reverse proxy (Traefik, Nginx, or Cloud ALB).
   - Enforces strict memory and CPU limits.

---

## F. Migration Plan for Step 3

To safely prepare environments without breaking the completed Step 2.8 runtime, the implementation should proceed in the following sequence:

### Phase 1: Core Configuration Refactoring (Step 3.2)
1. Decouple `is_production()` from `DATABASE_URL` in `backend/core/security.py`.
2. Add first-class `is_staging()` and support `ENVIRONMENT=staging` explicitly.
3. Add `environment` and database/Redis fields to `Settings` in `backend/config.py`.
4. Add `REDIS_PASSWORD` and `REDIS_SSL` support to `backend/services/cache_service.py` and `health_service.py`.

### Phase 2: Environment Templates & Secret Separation (Step 3.3)
1. Create `.env.development.example` (local SQLite/PostgreSQL defaults).
2. Create `.env.staging.example` (staging domain, staging DB names, staging key placeholders).
3. Create `.env.production.example` (production domain, KMS placeholders, HA DB configuration).
4. Ensure all `.env.*` files remain protected under `.gitignore`.

### Phase 3: Docker & Compose Restructuring (Step 3.4)
1. Maintain backward compatibility of `docker-compose.yml`.
2. Create standalone multi-stage production Dockerfile for frontend (`frontend/Dockerfile`) to eliminate `npm install && npm run build` at container boot.
3. Create `docker-compose.override.yml` for local development.
4. Create `docker-compose.prod.yml` for staging/production immutable deployment.

### Phase 4: Environment Validation & Verification (Step 3.5)
1. Verify DEV profile (local execution + Docker dev overlay).
2. Verify STAGING profile (simulated cloud staging with non-localhost CORS and strict keys).
3. Verify PROD profile (simulated production mode with full fail-fast checks).
4. Update `docs/DEVOPS_CHECKPOINT.md` to seal Step 3.

---

## G. Audit Sign-Off

* **Read-Only Audit Status:** 🟢 **COMPLETE**
* **Step 2.8 State:** **UNCHANGED & VERIFIED**
* **Application Code:** **UNTOUCHED**
* **Ready for Step 3 Implementation Planning.**
