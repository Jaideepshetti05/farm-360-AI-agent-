# Farm360 AI — Prioritized Technical Backlog (`TODO.md`)

> **Version:** 3.1.0  
> **Generated Date:** 2026-07-30  
> **Last Updated:** 2026-07-30  
> **Repository Commit:** cf67062  
> **Generator:** Antigravity AI Architecture Engine  
> **Status:** Active / Categorized Task List  

---

## 🔴 Critical Priority (P0 — Security & Secret Leakage)

- [x] **Next.js Server API Proxy — ✅ COMPLETED (v3.1.1):**
  - *Location:* `frontend/src/` — `Sidebar.tsx`, `VisionUpload.tsx`, `.env.local`, all 4 `/api/*/route.ts` files
  - *Issue:* `NEXT_PUBLIC_FARM360_API_KEY` exposed backend secret in client JS bundles.
  - *Resolution:* `VisionUpload` now calls `/api/vision-predict?task=` server proxy (no `apiKey` prop). `NEXT_PUBLIC_FARM360_API_KEY` removed from `.env.local` and all fallback chains. Build verified clean.
- [ ] **Redis-Backed Rate Limiting:**
  - *Location:* `backend/app.py` (lines 51-71)
  - *Issue:* Current `RateLimiter` is stored in process memory; fails under multi-worker Uvicorn deployment.
  - *Fix:* Replace in-memory dictionary with Redis sliding window counter.

---

## 🟠 High Priority (P1 — Performance & Scaling)

- [ ] **Async SSE Queue Iteration:**
  - *Location:* `backend/app.py` (lines 377-389)
  - *Issue:* Spawns a dedicated OS thread (`threading.Thread`) per SSE connection.
  - *Fix:* Refactor to native async generator queue iteration (`asyncio.Queue`).
- [ ] **Async File I/O for Image Uploads:**
  - *Location:* `backend/app.py` (lines 471-473)
  - *Issue:* `shutil.copyfileobj` blocks the event loop during file uploads.
  - *Fix:* Implement `aiofiles` for non-blocking file streaming.
- [ ] **Memory Session Pruning & TTL:**
  - *Location:* `backend/memory/session.py` (lines 39-46)
  - *Issue:* In-memory session dictionary grows indefinitely without eviction.
  - *Fix:* Implement LRU cache with 24-hour TTL eviction policy.

---

## 🟡 Medium Priority (P2 — Reliability & Features)

- [ ] **RAG Agricultural Document Batch Loader:**
  - *Location:* `backend/rag/loader.py`
  - *Issue:* PDF and Markdown document parsing script is not fully hooked to batch ingestion.
  - *Fix:* Build CLI batch ingestion script to populate `document_chunks` with 768-dim embeddings.
- [ ] **Weather Client Fallback Warning:**
  - *Location:* `backend/external_apis/weather.py` (lines 12-18)
  - *Issue:* Weather API silently returns mock values when `OPENWEATHER_API_KEY` is absent.
  - *Fix:* Log explicit warnings and indicate mock state in returned payload.
- [ ] **Blob URL Memory Leak Fix:**
  - *Location:* `frontend/src/components/ChatInput.tsx` (line 91)
  - *Issue:* `URL.createObjectURL()` is called without matching `URL.revokeObjectURL()`.
  - *Fix:* Add `revokeObjectURL` in component unmount / image submit handler.

---

## 🟢 Low Priority (P3 — Code Cleanup & Refactoring)

- [ ] **Legacy Config Field Cleanup:**
  - *Location:* `backend/config.py` (line 20)
  - *Issue:* Unused legacy `google_api_key` field remains in Pydantic Settings.
  - *Fix:* Deprecate legacy key fields and rely on `provider_manager` pool parsing.
- [ ] **Standardize Logging Output:**
  - *Location:* `backend/main.py` (lines 72, 76, 80)
  - *Issue:* Uses `sys.stdout.buffer.write()` mixed with Loguru logs.
  - *Fix:* Replace stdout buffer calls with standard `logger.debug()` calls.

---

## 🔵 Future & Research Roadmap

- [ ] **ONNX Model Quantization:** Quantize PyTorch ResNet18 vision models to INT8 for mobile/edge TPU deployment.
- [ ] **Drone & Satellite Imagery Ingestion:** Build multi-spectral image parsing pipeline for NDVI vegetation index calculations.
- [ ] **Multi-Agent Farm Simulation:** Build autonomous agent loops where CropAdvisor and LivestockAdvisor cross-examine recommendations.
