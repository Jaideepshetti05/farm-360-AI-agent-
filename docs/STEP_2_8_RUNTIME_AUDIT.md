# Farm360 DevOps — Step 2.8: Production Packaging & Container Runtime Verification Audit

**Audit Date:** 2026-09-02  
**Audit Type:** Comprehensive Read-Only Runtime Readiness Audit  
**Phase:** Step 2.8 DevOps Production Remediation  
**Status:** **AUDIT COMPLETE (READ-ONLY) — DO NOT MODIFY CODE YET**

---

## 1. Executive Summary

This audit assesses the runtime readiness of the Farm360 AI agricultural advisory platform for single-command (`docker compose up`) containerized deployment. While the core Python backend, Next.js frontend, and Alembic database migration foundations are verified, several container-orchestration gaps must be remediated before the full stack can start and operate in a production-like multi-container topology.

---

## 2. Comprehensive Subsystem Findings

### 2.1 Docker Engine Status
* **Docker Client Version:** `29.6.2` (API `1.55`, Go `1.26.5`, Windows/amd64)
* **Docker Compose Version:** `v5.3.1`
* **Docker Linux Daemon Status:** **BLOCKED / OFFLINE** (Named pipe `//./pipe/dockerDesktopLinuxEngine` unavailable; Docker Desktop is stopped on host)
* **Status:** `BLOCKED` for live container execution; `PASS` for static configuration & compilation validation.

---

### 2.2 Dockerfile & Build Readiness
* **Base Image:** `python:3.11-slim`
* **System Packages:** Installs `libgl1`, `libglib2.0-0`, `gcc`
* **Build Context Exclusions:** Root [`.dockerignore`](file:///c:/Users/Jaideep/Desktop/ml%20models/.dockerignore) excludes `.git`, `venv`, `__pycache__`, `node_modules`, `clean.zip` (1.9 GB), and local logs.
* **Pre-runtime Verification:** Dockerfile verifies `from google import genai`, `import torch`, and `from loguru import logger`.
* **Discrepancy / Risk:** `curl` is **not installed** in `Dockerfile`, but `docker-compose.yml` specifies a container healthcheck using `curl -f http://localhost:8000/`.
* **Status:** `FAIL` (Missing `curl` package in backend image for healthcheck probe).

---

### 2.3 Docker Compose Architecture
* **Declared Services:**
  1. `farm360-backend` (FastAPI, Port `8000:8000`)
  2. `farm360-frontend` (Next.js 16 / Node 20, Port `3000:3000`)
* **Missing Services in `docker-compose.yml`:**
  * ❌ **PostgreSQL Service** (e.g. `pgvector/pgvector:pg16` or `postgres:16-alpine`) is **absent**.
  * ❌ **Redis Service** (e.g. `redis:7-alpine`) is **absent**.
* **Missing Backend Environment Variables:**
  * `DATABASE_URL` (currently unpassed; falls back to ephemeral container SQLite)
  * `CORS_ORIGINS` (required if `ENVIRONMENT=production`)
  * `FARM360_ENCRYPTION_KEY` (required in production)
  * `REDIS_HOST` / `REDIS_PORT`
  * `ENVIRONMENT` / `APP_ENV`
* **Status:** `FAIL` (Incomplete production service topology).

---

### 2.4 Database & Migrations
* **PostgreSQL + pgvector Readiness:** Alembic migration [`migrations/versions/0001_initial_schema.py`](file:///c:/Users/Jaideep/Desktop/ml%20models/migrations/versions/0001_initial_schema.py) compiles cleanly for both PostgreSQL and SQLite.
* **Dialect Safety:** `CREATE EXTENSION IF NOT EXISTS vector;` executed conditionally on PostgreSQL; falls back to `Text` on SQLite.
* **Application Lifespan:** `backend/app.py` retains `create_all()` fallback for local SQLite.
* **Status:** `PASS` (Schema foundation verified; pending PostgreSQL container in Compose).

---

### 2.5 Redis & Caching
* **Implementation:** [`backend/services/cache_service.py`](file:///c:/Users/Jaideep/Desktop/ml%20models/backend/services/cache_service.py) provides Level 1 in-memory cache + Level 2 Redis cache.
* **Resilience:** Implements exponential backoff quarantine (`offline_until`) so cache misses/disconnections do not crash backend request threads.
* **Status:** `PASS` (Backend handles absent Redis gracefully, but Redis service should be in Compose for production).

---

### 2.6 ML Model Inventory & Volume Mounts
All 4 primary models required by [`backend/config.py`](file:///c:/Users/Jaideep/Desktop/ml%20models/backend/config.py) exist on host disk and are mapped into container volumes:

| Model Category | File / Directory Path | Size | Mount Target | Status |
|---|---|---|---|---|
| **Crop Regression** | `machine_learning/crop_regression/models/production_model_log.pkl` | 0.17 MB | `/app/machine_learning/crop_regression/models` | 🟢 `PASS` |
| **Dairy Intelligence** | `machine_learning/models/dairy_intelligence_v1_20260217_210257.pkl` | 0.003 MB | `/app/machine_learning/models` | 🟢 `PASS` |
| **Animal Disease** | `machine_learning/models/animal_disease_20260218_215356/` (25 model files) | ~504.0 MB | `/app/machine_learning/models` | 🟢 `PASS` |
| **Crop Vision** | `machine_learning/crop_vision/models/crop_disease_model.pth` | 42.74 MB | `/app/machine_learning/crop_vision/models` | 🟢 `PASS` |

---

### 2.7 Frontend Runtime & Server Proxy
* **Next.js Version:** `16.2.0` (React `19.2.4`)
* **Build Command:** `npm install && npm run build && npm start`
* **Server-Side API Proxies:**
  * [`frontend/src/app/api/chat/route.ts`](file:///c:/Users/Jaideep/Desktop/ml%20models/frontend/src/app/api/chat/route.ts) — Node.js runtime, forwards `X-API-Key`.
  * [`frontend/src/app/api/chat-stream/route.ts`](file:///c:/Users/Jaideep/Desktop/ml%20models/frontend/src/app/api/chat-stream/route.ts) — Real-time SSE streaming with `X-Accel-Buffering: no` header.
  * [`frontend/src/app/api/analyze-image/route.ts`](file:///c:/Users/Jaideep/Desktop/ml%20models/frontend/src/app/api/analyze-image/route.ts)
  * [`frontend/src/app/api/vision-predict/route.ts`](file:///c:/Users/Jaideep/Desktop/ml%20models/frontend/src/app/api/vision-predict/route.ts)
* **Secret Protection:** Zero client bundle leaks; `NEXT_PUBLIC_API_BASE_URL` points to `http://farm360-backend:8000` via Compose internal DNS.
* **Status:** `PASS`.

---

### 2.8 Application Endpoints & Verification Target
The following endpoints constitute the validation test surface for container runtime:

1. `GET /` — API Gateway Status
2. `GET /health/liveness` — Container liveness probe
3. `GET /health/readiness` — Deep health check (DB, Redis, models)
4. `GET /vision/health` — Vision service router status
5. `GET /vision/models` — Manifest of 6 vision models
6. `GET /api/health/providers` — Provider status (Gemini native, OpenRouter)
7. `POST /chat_stream` (with `X-API-Key`) — Agricultural advisory LLM stream
8. `POST /vision/crop-disease/predict` — Computer vision leaf diagnosis

---

### 2.9 Resource Sizing Requirements

| Subsystem | RAM (Min) | RAM (Recommended) | Storage | CPU Cores |
|---|---|---|---|---|
| **Backend (FastAPI + PyTorch + Scikit-Learn)** | 2.0 GB | 4.0 GB | 2.0 GB (Image) + 600 MB (Models) | 2 Cores |
| **Frontend (Next.js SSR / Node 20)** | 500 MB | 1.0 GB | 800 MB (Image) | 1 Core |
| **PostgreSQL + pgvector** | 250 MB | 500 MB | 5.0 GB (Data Volume) | 1 Core |
| **Redis** | 50 MB | 128 MB | 500 MB | 1 Core |
| **Total Production Stack** | **3.0 GB** | **6.0 GB - 8.0 GB** | **~10.0 GB** | **2 - 4 Cores** |

---

## 3. Detailed Audit Matrix

| Check / Requirement | Status | Detailed Finding |
|---|---|---|
| 1. Docker Engine Availability | 🟡 **BLOCKED** | Docker Desktop stopped on host; live execution blocked |
| 2. Dockerfile Build Compatibility | 🔴 **FAIL** | `curl` missing from `python:3.11-slim` build image for healthcheck |
| 3. Dockerfile Import Verification | 🟢 **PASS** | `from google import genai`, `import torch`, `from loguru import logger` pass |
| 4. `.dockerignore` Context Protection | 🟢 **PASS** | Excludes 1.9 GB zip, `.git`, `venv`, logs, and node_modules |
| 5. Compose Backend ↔ Frontend Network | 🟢 **PASS** | Shared default network resolves `farm360-backend:8000` |
| 6. Compose PostgreSQL Service | 🔴 **FAIL** | PostgreSQL container missing from `docker-compose.yml` |
| 7. Compose Redis Service | 🟠 **HIGH** | Redis container missing from `docker-compose.yml` |
| 8. Compose Secret & Fallback Hygiene | 🟢 **PASS** | Zero hardcoded API keys in compose file |
| 9. Alembic PostgreSQL Migration Script | 🟢 **PASS** | 18 tables, vector extension, indexes compiled cleanly |
| 10. ML Model Host File Existence | 🟢 **PASS** | All 4 production models present and correctly mounted |
| 11. Backend CORS Enforcement | 🟢 **PASS** | `_resolve_cors_origins()` rejects wildcards in production |
| 12. Frontend Server Proxy Protection | 🟢 **PASS** | All 4 API proxy routes protect API keys and disable SSE buffering |
| 13. Local SQLite Non-Regression | 🟢 **PASS** | Zero-setup development and unit test suites pass |

---

## 4. Issues & Priority Classification

### 🔴 Critical Blockers (Must fix for full Docker stack operation)
1. **Missing `curl` in Dockerfile:** The backend container healthcheck (`curl -f http://localhost:8000/`) will fail and mark the container unhealthy unless `curl` is added to `apt-get install` in `Dockerfile`.
2. **Missing PostgreSQL Service in `docker-compose.yml`:** The Compose file does not define a database container, leaving production deployments without persistent relational storage.

### 🟠 High Priority Issues
1. **Missing Redis Service in `docker-compose.yml`:** Cache service operates in fallback in-memory mode without a centralized Redis node.
2. **Missing Production Environment Injections:** `docker-compose.yml` backend service should explicitly pass `DATABASE_URL`, `CORS_ORIGINS`, `REDIS_HOST`, and `FARM360_ENCRYPTION_KEY`.

### 🟡 Medium Issues
1. **Next.js Build in Development Container:** Compose currently runs `npm install && npm run build && npm start` inside an un-cached container volume on startup rather than utilizing a multi-stage production frontend Dockerfile.

---

## 5. Recommended Remediation Order for Step 2.8

1. **Phase 1: Dockerfile Polish** — Add `curl` to `apt-get install` in `Dockerfile` for healthcheck reliability.
2. **Phase 2: Docker Compose Stack Completion** — Add `postgres` (`pgvector/pgvector:pg16`) and `redis` (`redis:7-alpine`) services with named volumes and complete environment injection.
3. **Phase 3: Database Migration Automation** — Add automated `alembic upgrade head` execution on backend container startup before launching Uvicorn.
4. **Phase 4: Runtime Stack Validation** — Validate static Compose compilation and test container boot when Docker Desktop engine is started.

---

*Step 2.8 audit complete. No files modified.*
