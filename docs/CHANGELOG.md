# Farm360 AI — Project Changelog (`CHANGELOG.md`)

> **Version:** 3.1.0  
> **Generated Date:** 2026-07-30  
> **Last Updated:** 2026-07-30  
> **Repository Commit:** cf67062  
> **Generator:** Antigravity AI Architecture Engine  
> **Status:** Active / Release History  

---

## [3.1.1] - 2026-07-30 — Security Patch: API Key Exposure Eliminated

### Security
- **P0 Fix — Client Bundle Secret Removed:** `NEXT_PUBLIC_FARM360_API_KEY` removed from `.env.local` and all client-side components. API key is now exclusively read from server-side `FARM360_API_KEY` environment variable.
- **VisionUpload Proxy Refactor:** `VisionUpload.tsx` no longer accepts `apiKey`/`backendUrl` props or calls the FastAPI backend directly. All vision requests now route through `/api/vision-predict?task=` Next.js server route which attaches the secret key server-side.
- **Server Route Hardening:** Removed `NEXT_PUBLIC_FARM360_API_KEY` fallbacks from all 4 server proxy routes (`/api/chat`, `/api/chat-stream`, `/api/analyze-image`, `/api/vision-predict`).
- **Confirmed Clean Build:** `npm run build` passes TypeScript type-check with zero errors. No `FARM360` string present in `.next/static/` client JS bundles.

### Fixed
- `ChatInput.tsx` L240: TypeScript strict null check for `res.body` now guarded with explicit null assertion throwing a descriptive error.
- `VisionUpload.tsx`: TypeScript strict mode errors resolved (`urgencyColor` typed as `Record<string, string>` with nullish fallback; `result.extra.quick_treatment` guarded with `!= null`).

---

## [3.1.0] - 2026-07-30

### Added
- **19-File Documentation Suite:** Complete documentation bootstrap and refinement in `docs/` (`AGENT_HANDOFF.md`, `TRACEABILITY_MATRIX.md`, `REPOSITORY_STRUCTURE.md`, `DOCUMENTATION_MANIFEST.md`, etc.).
- **Empirical Code Statistics:** Calculated exact line counts (~13,732 clean source LOC across 180 Python and 12 TypeScript files).
- **Readiness & Liveness Endpoints:** Added `/health/readiness` and `/health/liveness` to `app.py`.
- **Vision Health Endpoint:** Added `/vision/health` exposing CPU/GPU memory usage, uptime, and request metrics.
- **Key Pool Diagnostics:** Added `/keys/status` endpoint for monitoring multi-key pool health.

### Security
- **Path Traversal Fix:** Refactored image upload to use UUID-based filenames (`sanitize_filename`).
- **Constant-Time Verification:** Replaced equality checks with `secrets.compare_digest` in API key verification.
- **Fernet PII Encryption:** Symmetrically encrypted `email` and `gps_coordinates` in database models (`EncryptedString`).

---

## [3.0.0] - 2026-07-06

### Added
- **Multi-Provider Key Manager:** Round-robin key rotation and 60-second quarantining across Google Gemini, OpenRouter, and OpenAI key pools.
- **Native Gemini SDK:** Upgraded backend to `google-genai` native SDK v1.x (`gemini-2.5-flash`).
- **Memory v2 Engine:** Recency-, importance-, and vector-ranked memory store with PostgreSQL ORM `UnitOfWork` and local JSON fallback.
- **Modular Vision Service:** 6 PyTorch ResNet18 vision routers (`crop-disease`, `breed`, `weed`, `detect`, `plant-id`, `fruit-grade`).

---

## [2.5.0] - 2026-06-22

### Added
- **Real-Time SSE Streaming:** Added `/chat_stream` POST endpoint emitting line-delimited `data: [token]\n\n` SSE streams.
- **Next.js 16 Web UI:** Released React 18 chat interface (`ChatCanvas`, `ChatInput`, `Sidebar`, `VisionUpload`).

---

## [2.0.0] - 2026-05-15

### Added
- **FastAPI Core Gateway:** Replaced CLI test scripts with production FastAPI web server (`app.py`).
- **Unified ML Wrapper:** Integrated scikit-learn yield regression, time-series dairy forecasting, and animal health diagnosis into `FarmAPIWrapper`.

---

## [1.0.0] - 2025-04-24

### Added
- Initial release of trained machine learning models for crop yield prediction and crop disease classification.
