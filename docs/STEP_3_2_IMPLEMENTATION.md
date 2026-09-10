# Farm360 DevOps — Step 3.2 Implementation Log

**Milestone:** Step 3 — Prepare Environments (DEV → STAGING → PROD)  
**Task:** Step 3.2 — Core Configuration & Security Decoupling  
**Status:** 🟡 **IMPLEMENTED — AWAITING USER REVIEW & SIGN-OFF**  
**Date:** 2026-09-03  

---

## 1. Summary of Changes Made

### A. Environment Detection & Staging Support (`backend/core/security.py`)
* **Decoupled from PostgreSQL:** Removed lines checking if `DATABASE_URL` starts with `postgresql` or `postgres`. PostgreSQL usage no longer forces the application into production mode.
* **Strict Environment Resolution:** `is_production()` evaluates `ENVIRONMENT`, `APP_ENV`, `ENV`, or `settings.environment`. Accepted production tokens are: `production`, `prod`, `live`.
* **Added First-Class Staging:** Introduced `is_staging() -> bool` recognizing: `staging`, `stage`, `preprod`, `pre-production`.
* **Preserved Fail-Fast Production Hardening:** In production mode, `Encryptor` strictly enforces `FARM360_ENCRYPTION_KEY` and raises `ValueError` if missing, preventing insecure XOR fallback.

### B. First-Class Environment Settings (`backend/config.py`)
* Added explicit environment, database, Redis, and CORS fields to `Settings`:
  - `environment: str = "development"`
  - `farm360_api_key: str | None = None`
  - `farm360_encryption_key: str | None = None`
  - `database_url: str | None = None`
  - `redis_host: str = "127.0.0.1"`
  - `redis_port: int = 6379`
  - `redis_db: int = 0`
  - `redis_password: str | None = None`
  - `redis_ssl: bool = False`
  - `cors_origins: str | None = None`

### C. Redis Authentication & TLS Support (`backend/services/cache_service.py` & `backend/services/health_service.py`)
* **Dynamic Credentials & TLS:** Both services now support `REDIS_PASSWORD` and `REDIS_SSL` (`true`, `1`, `yes`, `on`).
* **Environment Variable Priority:** Runtime `os.environ` takes priority over static `settings` fields, followed by fallback defaults.
* **Socket Connect Timeout:** Added `socket_connect_timeout=1.0` to `health_service.py` ensuring fast-failing health checks instead of 45-second OS TCP timeouts on offline hosts.
* **Circular Import Resolution:** Scoped `StreamingConfig` import inside `CacheService.set()` to prevent circular import between `backend.services.cache_service` and `backend.streaming.__init__`.

---

## 2. Environment Semantics Matrix

| Environment | Setting (`ENVIRONMENT`) | `is_production()` | `is_staging()` | PII Encryption Rule | CORS Rule | Redis Auth / TLS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Development** | `development` (default) | `False` | `False` | Fallback XOR allowed with warning | Localhost origins allowed if unset | Optional (none by default) |
| **Staging** | `staging` / `preprod` | `False` | `True` | Non-production fallback allowed | Explicit staging origin | Supported (`REDIS_PASSWORD`, `REDIS_SSL`) |
| **Production** | `production` / `live` | `True` | `False` | Strict Fernet key required (fail-fast) | Strict explicit origins required (no `*`) | Supported (`REDIS_PASSWORD`, `REDIS_SSL`) |

---

## 3. Files Modified

| File | Scope of Edits |
| :--- | :--- |
| [`backend/core/security.py`](file:///c:/Users/Jaideep/Desktop/ml%20models/backend/core/security.py) | Decoupled `is_production()` from `DATABASE_URL`; added `is_staging()`. |
| [`backend/config.py`](file:///c:/Users/Jaideep/Desktop/ml%20models/backend/config.py) | Added environment, database, Redis, and CORS fields to `Settings`. |
| [`backend/services/cache_service.py`](file:///c:/Users/Jaideep/Desktop/ml%20models/backend/services/cache_service.py) | Added `password` and `ssl` to `redis.Redis`; scoped `StreamingConfig` import. |
| [`backend/services/health_service.py`](file:///c:/Users/Jaideep/Desktop/ml%20models/backend/services/health_service.py) | Added `password`, `ssl`, `db`, and `socket_connect_timeout=1.0` to Redis check. |

---

## 4. Test Suite Execution & Results

| # | Verification Check | Scope / Target | Result |
| :--- | :--- | :--- | :--- |
| **1** | `ENVIRONMENT=development` | `is_production() == False`, `is_staging() == False` | 🟢 **PASS** |
| **2** | `ENVIRONMENT=staging` | `is_production() == False` | 🟢 **PASS** |
| **3** | `ENVIRONMENT=staging` | `is_staging() == True` | 🟢 **PASS** |
| **4** | `ENVIRONMENT=production` | `is_production() == True`, `is_staging() == False` | 🟢 **PASS** |
| **5** | PostgreSQL + `ENVIRONMENT=development` | `is_production() == False` (decoupling verified) | 🟢 **PASS** |
| **6** | PostgreSQL + `ENVIRONMENT=staging` | `is_production() == False`, `is_staging() == True` | 🟢 **PASS** |
| **7** | Production without encryption key | Raises `ValueError` ("must be explicitly set in production mode") | 🟢 **PASS** |
| **8** | DEV Redis configuration | `password=None`, `ssl=False` passed to `redis.Redis` | 🟢 **PASS** |
| **9** | STAGING / PROD Redis password | `password='super-secret-redis-password'` passed to `redis.Redis` | 🟢 **PASS** |
| **10**| STAGING / PROD Redis SSL/TLS | `ssl=True` passed to `redis.Redis` | 🟢 **PASS** |
| **11**| HealthService Redis credentials & TLS | `password`, `ssl=True`, `db=1` correctly passed | 🟢 **PASS** |
| **12**| Secret Scanning | Zero hardcoded passwords, tokens, or encryption keys in source | 🟢 **PASS** |
| **13**| Existing Unit Tests | `backend/tests/test_stream.py` (10 tests) | 🟢 **PASS** (10/10) |
| **14**| Python Syntax & Compilation | `py_compile` across all 4 modified files | 🟢 **PASS** |

---

## 5. Known Limitations
* **Step 3.2 is confined to core decoupling**: Environment templates (`.env.dev.example`, `.env.staging.example`, `.env.prod.example`) and Compose overlays are scheduled for subsequent sub-steps (Steps 3.3 and 3.4).
* **LLM Provider Health**: Reports `offline` in local test environments because provider API keys are not configured.
