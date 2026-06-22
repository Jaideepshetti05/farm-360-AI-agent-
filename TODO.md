# Farm360 AI Codebase — Issues & Improvement Plan

## 1. HARDCODED SECRETS & CONFIGURATION ISSUES

### 1.1 Hardcoded API Key in Backend
**File:** `backend/config.py` (line 8)
**Issue:** `farm360_api_key: str = "secure-secret-key-1234"` — a static, hardcoded secret. Bad for security, no rotation possible.
**Fix:** Load from environment variable with a random fallback; log a warning if using default.

### 1.2 Hardcoded API Key in Frontend
**File:** `frontend/src/components/ChatInput.tsx` (line 106)
**Issue:** `const apiKey = process.env.NEXT_PUBLIC_FARM360_API_KEY || "secure-secret-key-1234"` — exposes the key to all frontend clients (public env var).
**Fix:** Use a non-prefixed env var that is server-side only, or make the `/chat` endpoints not require a public-facing API key at all since CORS is set to `*`.

### 1.3 Default API Key in app.py
**File:** `backend/app.py` (line 68-72)
**Issue:** The API key verification uses the same hardcoded default. Anyone who knows this key can access the API.
**Fix:** Load from env, use strong random key generation as fallback with warning.

### 1.4 CORS: Allow-Origins "*" with Credentials False
**File:** `backend/app.py` (lines 57-63)
**Issue:** `allow_origins=["*"]` with `allow_credentials=False`. While this is technically safe (no credentials + wildcard), it's overly permissive for production.
**Fix:** Restrict to specific frontend domain in production.

---

## 2. CODE QUALITY & ROBUSTNESS ISSUES

### 2.1 Config uses `google_api_key` but code never uses it
**File:** `backend/config.py` (line 6), `backend/main.py` (line 64)
**Issue:** `google_api_key` is defined but the actual LLM client uses `openrouter_api_key`. The test script `test_llm.py` tests Google Gemini, while the actual app uses OpenRouter. Mismatched / dead config.
**Fix:** Remove `google_api_key` from config or add a note that it's legacy/unused.

### 2.2 Model paths in config are relative but may not exist
**File:** `backend/config.py` (lines 11-14)
**Issue:** Hardcoded paths like `"crop_regression/models/production_model_log.pkl"` — no validation on startup that these files exist.
**Fix:** Add startup validation with clear error messages logged.

### 2.3 Silent Exception Catching in MediaPipeline
**File:** `backend/media_pipeline/image_processor.py` (line 22-23)
**Issue:** `try: ... except NameError: self.transform = None` — silently swallows import errors (e.g., if torch/PIL not installed). The component will fail at runtime with a cryptic error.
**Fix:** Log the exception clearly and re-raise or provide a meaningful fallback.

### 2.4 No Input Validation for LLM responses in chat endpoints
**File:** `backend/app.py` (lines 163-179, 183-206)
**Issue:** The `/chat` and `/analyze_image` endpoints return LLM output directly without sanitization or validation for harmful/prohibited content.
**Fix:** Add output filtering layer.

### 2.5 Duplicate Module Loading / Circular Import Risk
**File:** `backend/main.py` (line 18) and `backend/app.py` (line 18)
**Issue:** Both import from `backend.main`. The import chain is: `app.py → main.py → model_wrapper.py → config.py`. This creates tight coupling.
**Fix:** Consider restructuring to avoid deep import chains.

### 2.6 LLMValidator class is unused in the streaming code path
**File:** `backend/api_gateway/model_wrapper.py` (class `LLMValidator`, lines 150-176)
**Issue:** `LLMValidator` exists for structured JSON validation but the actual system prompt (main.py line 34) says "never output raw JSON". The validator is dead code — defined but never called.
**Fix:** Either remove it or repurpose it for the non-streaming endpoint.

---

## 3. SECURITY VULNERABILITIES

### 3.1 Path Traversal Vulnerability in Image Upload
**File:** `backend/app.py` (lines 192-193)
**Issue:** `temp_path = os.path.join(TEMP_DIR, image.filename)` — the user-supplied filename could contain `../` to escape `temp_uploads/`. An attacker could overwrite arbitrary files.
**Fix:** Use a secure filename sanitizer (e.g., uuid-based naming).

### 3.2 No Rate Limiting
**File:** `backend/app.py` — no middleware
**Issue:** No rate limiting on any endpoint. An attacker could hammer the LLM API indefinitely, racking up costs.
**Fix:** Add `slowapi` middleware or a custom rate limiter.

### 3.3 No Input Size Limits
**File:** `backend/app.py`
**Issue:** The `query` string and `image` upload have no maximum size validation. Could DoS the server with large payloads.
**Fix:** Add `max_length` on query, max file size on uploads.

### 3.4 No HTTPS enforcement
**File:** `backend/app.py` (line 211)
**Issue:** Runs on plain HTTP with no TLS. API keys transmitted in cleartext.
**Fix:** Use a reverse proxy (nginx/caddy) with TLS.

---

## 4. PERFORMANCE & SCALABILITY

### 4.1 Thread-per-Request for Streaming (Blocking I/O)
**File:** `backend/app.py` (lines 120-134)
**Issue:** Each SSE connection spawns a dedicated `threading.Thread`. Under load (>100 concurrent users), this will exhaust thread pools and degrade performance.
**Fix:** Use `asyncio.to_thread()` or an async generator approach to avoid unbounded thread creation.

### 4.2 Synchronous File I/O in Event Loop
**File:** `backend/app.py` (lines 196-198, 206-207)
**Issue:** `open`, `shutil.copyfileobj`, `os.remove` are blocking I/O calls in async endpoints.
**Fix:** Wrap in `run_in_threadpool` or use `aiofiles`.

### 4.3 Synchronous Image Loading in MediaPipeline
**File:** `backend/media_pipeline/image_processor.py` (lines 31-33)
**Issue:** `Image.open()` and `.convert()` are blocking CPU-bound operations.
**Fix:** Offload to thread pool or make async.

### 4.4 No connection pooling for HTTP
**File:** `backend/external_apis/weather.py` (line 27)
**Issue:** Creates a new `requests.get()` call each time. No connection reuse.
**Fix:** Use `httpx.AsyncClient` with connection pooling.

### 4.5 Memory grows unbounded with session history
**File:** `backend/memory/session.py` (lines 39-46)
**Issue:** Sessions stored in-memory dict. After many conversations, memory usage grows indefinitely. No TTL/eviction policy.
**Fix:** Implement LRU eviction, set max sessions, add TTL expiry.

### 4.6 JSON save/load on every message
**File:** `backend/memory/session.py` (line 54, 55)
**Issue:** Every `add_message()` triggers a full JSON serialization + disk write of the entire memory file. High I/O overhead.
**Fix:** Batch writes, use append-only log, or switch to SQLite/Redis.

---

## 5. ERROR HANDLING & RELIABILITY

### 5.1 Token Stream: Error in producer leaves stream hanging
**File:** `backend/app.py` (lines 122-134)
**Issue:** If `token_queue.get` blocks indefinitely (e.g., exception before `None` sentinel), the SSE stream hangs forever (no timeout).
**Fix:** Add timeout on queue.get, catch exceptions in the async loop.

### 5.2 No retry logic for LLM API calls
**File:** `backend/main.py` (lines 153-163)
**Issue:** If OpenRouter API returns a transient error (5xx, rate limit), the stream fails immediately with a fallback.
**Fix:** Add automatic retry with exponential backoff for transient failures.

### 5.3 Fallback text shown as token stream error
**File:** `backend/app.py` (line 128)
**Issue:** Error messages from producer are sent into the token stream as raw text. No distinction between "system error" and "LLM content."
**Fix:** Use a structured error event format (e.g., `data: [ERROR]...`).

### 5.4 Thread safety: global `agent` object mutated without lock
**File:** `backend/app.py` (line 22, 31-44)
**Issue:** The `agent` global is set during startup. If the app had hot-reload paths that re-assign it, concurrent requests could see inconsistent state. Currently not a problem with single-assignment, but fragile.
**Fix:** Use a module-level lock or dependency injection.

---

## 6. CONFIGURATION & DEPLOYMENT

### 6.1 Path Resolution in Docker vs Local is Fragile
**File:** `backend/api_gateway/model_wrapper.py` (line 23)
**Issue:** `docker_mount_base = "/app" if os.path.exists("/app") and os.path.isdir("/app") else BASE_DIR` — fragile heuristic. Could misdetect in CI environments.
**Fix:** Use `settings` env variable (`MODEL_BASE_PATH`) to explicitly control.

### 6.2 .env file not tracked or documented
**Issue:** There's no `.env.example` file. Need to reverse-engineer required env vars from config.py (OPENROUTER_API_KEY, FARM360_API_KEY, OPENWEATHER_API_KEY).
**Fix:** Create `.env.example` with all required variables documented.

### 6.3 Python version not pinned
**File:** `Dockerfile` (needs checking)
**Issue:** No `.python-version` file or explicit pin.
**Fix:** Pin Python version and add `.python-version` / `runtime.txt`.

### 6.4 No health check for model availability
**File:** `backend/app.py` (lines 76-83)
**Issue:** Health check returns `"llm_active": agent.has_llm` but doesn't test that models are actually loaded. Could report healthy while models failed to load.
**Fix:** Add model file existence checks to health endpoint.

---

## 7. FRONTEND ISSUES

### 7.1 Stream never fully consumed on error
**File:** `frontend/src/components/ChatInput.tsx` (lines 170-201)
**Issue:** The SSE reader could leak if the stream encounters an error mid-way through parsing (e.g., malformed SSE frame). The `reader.cancel()` is only called on `[DONE]`.
**Fix:** Add a `finally` block to always cancel the reader.

### 7.2 Preview URL never revoked
**File:** `frontend/src/components/ChatInput.tsx` (line 91)
**Issue:** `URL.createObjectURL(submittedFile)` creates a blob URL that is never revoked with `URL.revokeObjectURL()`. Memory leak.
**Fix:** Call `revokeObjectURL` when the message is no longer displayed.

### 7.3 API Key exposed in client bundle
**File:** `frontend/src/components/ChatInput.tsx` (line 106)
**Issue:** `NEXT_PUBLIC_FARM360_API_KEY` is exposed to browser. Anyone can inspect the source and see it.
**Fix:** Use a Next.js API route as proxy for LLM calls, keep key server-side.

### 7.4 No CSRF protection
**File:** `frontend/src/components/ChatInput.tsx`
**Issue:** The frontend sends direct POST requests to the backend. If the backend is accessed from a malicious site, it could make cross-origin requests (though CORS `*` mitigates this).
**Fix:** Use session-based tokens or `SameSite` cookies.

---

## 8. MISCELLANEOUS

### 8.1 Inconsistent encoding: `sys.stdout.buffer.write()` used for debug output
**File:** `backend/main.py` (lines 72, 76, 80)
**Issue:** Direct byte-level stdout writes mixed with `logger` — awkward, breaks structured logging, and could interfere with production logging.
**Fix:** Remove and use only `logger` for all diagnostics.

### 8.2 Undefined `get_session_id()` in frontend
**File:** `frontend/src/components/ChatInput.tsx` (line 113)
**Issue:** `session_id` hardcoded as `"default_react_session"` — no session isolation between users. All conversations share the same history.
**Fix:** Generate a unique session ID on page load (stored in localStorage).

### 8.3 WeatherClient returns mocked data silently
**File:** `backend/external_apis/weather.py` (lines 12-18)
**Issue:** If `OPENWEATHER_API_KEY` is missing, it returns hardcoded fake values without logging a warning. The fake `rain_chance: 85` is suspiciously high.
**Fix:** Log a warning when using mock data or fail explicitly.

### 8.4 Dockerfile needs review
**Issue:** There's a `Dockerfile` and `docker-compose.yml` at root, but the model paths and dependencies may not work in Docker.
**Fix:** Review and test the Docker build end-to-end.

### 8.5 Thread-safety of `_save()` not guaranteed
**File:** `backend/memory/session.py` (lines 31-37)
**Issue:** While `_save()` acquires the lock, JSON serialization and file write are not atomic. If two threads call `_save()` concurrently, the file could be corrupted.
**Fix:** Write to a temp file first, then rename (atomic on most OS).

---

## PRIORITY ACTION ITEMS

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| 🔴 P0 | Path traversal in image upload | Low | Critical (security) |
| 🔴 P0 | Hardcoded secrets | Low | Critical (security) |
| 🔴 P0 | API key exposed in frontend bundle | Medium | Critical (security) |
| 🟠 P1 | No rate limiting | Low | High |
| 🟠 P1 | Thread-per-request scaling | Medium | High |
| 🟠 P1 | Memory leak in session storage | Medium | High |
| 🟡 P2 | Blocking I/O in async endpoints | Medium | Medium |
| 🟡 P2 | No error distinction in stream | Low | Medium |
| 🟡 P2 | Blob URL never revoked (frontend) | Low | Medium |
| 🟢 P3 | LLMValidator dead code | Low | Low |
| 🟢 P3 | Config cleanup (google_api_key) | Low | Low |
| 🟢 P3 | .env.example creation | Low | Low |