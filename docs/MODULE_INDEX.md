# Farm360 AI — Comprehensive Module Index (`MODULE_INDEX.md`)

> **Version:** 3.1.1  
> **Generated Date:** 2026-07-30  
> **Last Updated:** 2026-07-30  
> **Repository Commit:** cf67062+security-patch  
> **Generator:** Antigravity AI Architecture Engine  
> **Status:** Active / Synchronized Module Catalog  

---

## 1. Executive Subsystem Index

| Module Name | Folder Path | Primary Responsibility | Current Status |
|---|---|---|---|
| **FastAPI Core Gateway** | `backend/` | Application routing, CORS, rate limiting, authentication | Operational |
| **Agent Orchestrator** | `backend/main.py` | Orchestrates LLM, memory, vision, & ML model inference | Operational |
| **Provider Manager** | `backend/provider_manager.py` | Multi-key round-robin rotation & multi-provider failover | Operational |
| **Computer Vision Service**| `backend/vision_service/` | Multi-task vision classification & Grad-CAM explainability | Operational |
| **Memory v2 Subsystem** | `backend/memory_v2/` | Recency-, importance-, & vector-ranked contextual store | Operational |
| **RAG Knowledge Engine** | `backend/rag/` | Document chunking, 768-dim embeddings, & retrieval | In Progress |
| **Intent Router** | `backend/router/` | Classifies query intent into domain advisor registries | Operational |
| **Guardrail Validator** | `backend/validator/` | Rule-based safety & policy validation engine | Operational |
| **Observability Engine** | `backend/observability/` | Asynchronous event bus and structured logging | Operational |
| **Streaming Manager** | `backend/streaming/` | SSE event formatting & stream metric collection | Operational |
| **Prompt Engine** | `backend/prompts/` | Jinja2 prompt template rendering | Operational |
| **Database Models** | `backend/models/` | SQLAlchemy ORM declarative models with Fernet encryption | Operational |
| **Machine Learning Models**| `machine_learning/` | PyTorch and scikit-learn models & training scripts | Operational |
| **Next.js 16 Web UI** | `frontend/` | React 18 frontend with real-time SSE streaming canvas | Operational |
| **Next.js Server API Proxies** | `frontend/src/app/api/` | Server-side route handlers attaching `FARM360_API_KEY` before proxying to FastAPI | Operational |

---

## 2. Exhaustive Module Specifications

### 2.1 Provider Manager Module (`backend/provider_manager.py`)
- **Purpose:** Handles zero-downtime multi-key rotation and multi-provider failover.
- **Responsibilities:** Ingest indexed environment keys (`GOOGLE_API_KEY_1...5`, `OPENROUTER_API_KEY_1...5`), quarantine rate-limited keys for 60 seconds, disable invalid keys, yield streaming tokens.
- **Dependencies:** `google-genai` (native SDK), `openai` (SDK), `loguru`.
- **Public APIs:** `provider_manager.stream_completion()`, `provider_manager.status()`, `provider_manager.health_status()`.
- **Key Classes:** `ProviderManager`, `KeyEntry`.
- **Key Functions:** `_get_gemini_client()`, `_get_openai_client()`, `mark_rate_limited()`.
- **Configuration:** Ingests keys from `.env` via `load_dotenv()`.
- **Current Status:** Operational (100% complete).
- **Related Documentation:** `docs/DECISIONS.md` (ADR-001, ADR-002), `docs/ARCHITECTURE.md`.
- **Future Improvements:** Add automated ping health checks for key pools.

---

### 2.2 Computer Vision Service (`backend/vision_service/`)
- **Purpose:** Provides multi-task computer vision REST APIs for crop disease diagnosis, breed identification, and fruit grading.
- **Responsibilities:** Load PyTorch model weights, preprocess image uploads, execute inference, compute Grad-CAM heatmaps, return structured JSON predictions.
- **Dependencies:** `torch`, `torchvision`, `Pillow`, `OpenCV`, `FastAPI`.
- **Public APIs:** `POST /vision/crop-disease`, `POST /vision/breed`, `POST /vision/weed`, `POST /vision/detect`, `POST /vision/plant-id`, `POST /vision/fruit-grade`, `GET /vision/health`.
- **Key Classes:** `VisionEngine`, `ModelRegistry`, `GradCAMExplainer`, `MetricsManager`.
- **Key Functions:** `predict()`, `explain()`, `sanitize_filename()`.
- **Configuration:** Reads model paths from `backend/config.py` (`crop_vision_model_path`).
- **Current Status:** Operational (ResNet18 crop disease active).
- **Related Documentation:** `docs/PROJECT_MEMORY.md`, `docs/ARCHITECTURE.md`.
- **Future Improvements:** Quantize PyTorch ResNet18 models to INT8 ONNX runtime.

---

### 2.3 Memory v2 Subsystem (`backend/memory_v2/`)
- **Purpose:** Manages context retrieval, scoring, and persistence for user conversation history.
- **Responsibilities:** Rank memory records using hybrid linear scoring (Recency + Importance + Frequency + Vector Similarity), write memories to PostgreSQL database, fall back to local JSON file.
- **Dependencies:** `SQLAlchemy`, `pydantic`, `numpy`, `loguru`.
- **Public APIs:** `memory_manager.add_memory()`, `memory_manager.get_relevant_memories()`.
- **Key Classes:** `MemoryStore`, `MemoryRanker`, `MemoryRetriever`, `MemoryScheduler`, `MemorySummarizer`, `MemoryRecordModel`.
- **Key Functions:** `rank_memories()`, `add_memory()`, `_save_fallback()`.
- **Configuration:** Writes to PostgreSQL database via `UnitOfWork` with fallback to `logs/memory_v2_fallback.json`.
- **Current Status:** Operational (85% complete).
- **Related Documentation:** `docs/DECISIONS.md` (ADR-006), `docs/ARCHITECTURE.md`.
- **Future Improvements:** Add automated memory compression scheduler worker.

---

### 2.4 RAG Engine Subsystem (`backend/rag/`)
- **Purpose:** Processes agricultural reference manuals into searchable vector chunks for grounded query answering.
- **Responsibilities:** Load PDF/Markdown documents, split into overlapping text chunks, generate 768-dim Gemini embeddings, compute cosine similarity retrieval and reranking.
- **Dependencies:** `google-genai`, `numpy`, `SQLAlchemy`.
- **Public APIs:** `rag_service.retrieve_context()`, `rag_service.ingest_document()`.
- **Key Classes:** `DocumentLoader`, `TextChunker`, `GeminiEmbedder`, `Retriever`, `Reranker`, `RAGService`.
- **Key Functions:** `chunk_text()`, `get_embedding()`, `retrieve()`.
- **Configuration:** Reads chunk size settings from `backend/rag/config.py`.
- **Current Status:** In Progress (60% complete).
- **Related Documentation:** `docs/PROJECT_MEMORY.md`, `docs/MASTER_ROADMAP.md`.
- **Future Improvements:** Complete CLI batch ingestion runner for ICAR manuals.

---

### 2.5 Intent Router Subsystem (`backend/router/`)
- **Purpose:** Classifies incoming user queries into specialized domain advisors.
- **Responsibilities:** Evaluate query keywords, route to `CropAdvisor`, `LivestockAdvisor`, `WeatherAdvisor`, or `GeneralAdvisor`.
- **Dependencies:** `FastAPI`, `pydantic`.
- **Public APIs:** `router.route_query()`.
- **Key Classes:** `IntentRouter`, `AdvisorRegistry`, `AdvisorResult`.
- **Key Functions:** `classify_intent()`, `get_advisor()`.
- **Status:** Operational.
- **Related Documentation:** `docs/ARCHITECTURE.md`.
- **Future Improvements:** Replace keyword classification with LLM zero-shot intent classifier.

---

### 2.6 Guardrail Validator Subsystem (`backend/validator/`)
- **Purpose:** Evaluates LLM responses and domain advice against agricultural safety rules.
- **Responsibilities:** Screen outputs for dangerous pesticide dosages, hallucinated chemical names, or forbidden raw JSON formats.
- **Dependencies:** `re`, `pydantic`.
- **Public APIs:** `validator.validate()`.
- **Key Classes:** `ValidationEngine`, `ValidationResult`.
- **Key Functions:** `validate_response()`, `check_safety_rules()`.
- **Status:** Operational.
- **Related Documentation:** `docs/PROJECT_MEMORY.md`.
- **Future Improvements:** Add automated toxicity and prompt injection detection rules.

---

### 2.7 Observability Subsystem (`backend/observability/`)
- **Purpose:** Provides structured logging, distributed trace context, and internal event dispatching.
- **Responsibilities:** Log events asynchronously via Loguru, propagate trace IDs, dispatch internal events via `EventBus`.
- **Dependencies:** `loguru`, `asyncio`.
- **Public APIs:** `event_bus.publish()`, `event_bus.subscribe()`.
- **Key Classes:** `EventBus`, `TraceContext`, `LoggerConfig`.
- **Status:** Operational.
- **Related Documentation:** `docs/PROJECT_MEMORY.md`.
- **Future Improvements:** Export trace metrics to OpenTelemetry collector.

---

### 2.8 Database Models Subsystem (`backend/models/database.py`)
- **Purpose:** Defines 18 SQLAlchemy declarative ORM models for database persistence.
- **Responsibilities:** Model tables (`User`, `UserProfile`, `ChatSession`, `ConversationHistory`, `MemoryRecord`, `Prediction`, `DocumentChunk`), handle Fernet AES-256 PII column encryption.
- **Dependencies:** `SQLAlchemy 2.0`, `Cryptography (Fernet)`.
- **Key Classes:** `EncryptedString`, `User`, `ChatSession`, `MemoryRecord`, `Prediction`, `DocumentChunk`.
- **Status:** Operational.
- **Related Documentation:** `docs/PROJECT_MEMORY.md`, `docs/DECISIONS.md` (ADR-008).
- **Future Improvements:** Generate Alembic database migration scripts.

---

### 2.9 Next.js Web UI Subsystem (`frontend/src/`)
- **Purpose:** User interface for real-time SSE streaming chat, vision uploads, and diagnostic display.
- **Responsibilities:** Render responsive chat canvas, parse SSE `data: [token]` streams, handle image dropzones, communicate with backend API via Next.js server proxy routes.
- **Dependencies:** Next.js 16, React 18, Tailwind CSS, TypeScript.
- **Key Components:** `ChatCanvas.tsx`, `ChatInput.tsx`, `Sidebar.tsx`, `VisionUpload.tsx`.
- **Status:** Operational ✅ (P0 security resolved in v3.1.1).
- **Related Documentation:** `docs/ARCHITECTURE.md`, `docs/AGENT_HANDOFF.md`, `docs/DECISIONS.md` (ADR-009).
- **Future Improvements:** Async SSE producer refactoring in `backend/app.py` (P1).

---

### 2.10 Next.js Server API Proxy Routes (`frontend/src/app/api/`)
- **Purpose:** Server-side route handlers that protect `FARM360_API_KEY` from client bundle exposure.
- **Responsibilities:** Receive browser form-data requests, read `FARM360_API_KEY` from Node.js server environment, attach as `X-API-Key` header, proxy to FastAPI backend, return response (streaming or JSON).
- **Dependencies:** Next.js 16 App Router, Node.js process environment.
- **Registered Routes:**
  | Route File | HTTP Method | Proxies To | Response Mode |
  |---|---|---|---|
  | `api/chat/route.ts` | POST | `POST /chat_stream` | SSE stream (unbuffered) |
  | `api/chat-stream/route.ts` | POST | `POST /chat_stream` | SSE stream (unbuffered) |
  | `api/analyze-image/route.ts` | POST | `POST /analyze_image` | JSON |
  | `api/vision-predict/route.ts` | POST | `POST /vision/{task}` | JSON |
- **Status:** Operational ✅ (created/hardened in v3.1.1).
- **Related Documentation:** `docs/DECISIONS.md` (ADR-009), `docs/ARCHITECTURE.md`.
- **Security:** `FARM360_API_KEY` confirmed absent from `.next/static/` client JS bundles.
