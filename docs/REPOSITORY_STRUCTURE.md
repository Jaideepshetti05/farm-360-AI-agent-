# Farm360 AI — Repository Structure & Directory Map (`REPOSITORY_STRUCTURE.md`)

> **Version:** 3.1.0  
> **Generated Date:** 2026-07-30  
> **Last Updated:** 2026-07-30  
> **Repository Commit:** cf67062  
> **Generator:** Antigravity AI Architecture Engine  
> **Status:** Active / Synchronized  

---

## 1. Directory Tree Overview

```
c:\Users\Jaideep\Desktop\ml models\
├── backend/
│   ├── agent_core/
│   ├── api_gateway/
│   ├── core/
│   ├── decision_engine/
│   ├── evaluation/
│   ├── external_apis/
│   ├── feedback/
│   ├── media_pipeline/
│   ├── memory/
│   ├── memory_v2/
│   ├── models/
│   ├── observability/
│   ├── performance/
│   ├── prompts/
│   │   └── templates/
│   ├── rag/
│   ├── repositories/
│   ├── router/
│   ├── services/
│   ├── streaming/
│   ├── tests/
│   ├── validator/
│   │   └── rules/
│   ├── vision_service/
│   │   ├── routes/
│   │   └── temp_uploads/
│   ├── app.py
│   ├── config.py
│   ├── main.py
│   └── provider_manager.py
├── frontend/
│   ├── public/
│   └── src/
│       ├── app/
│       │   └── api/
│       └── components/
├── machine_learning/
│   ├── animal_module/
│   ├── crop_regression/
│   ├── crop_vision/
│   ├── dairy_module/
│   ├── data/
│   ├── docs/
│   ├── experiments/
│   ├── health_module/
│   ├── model_registry/
│   ├── models/
│   ├── reports/
│   ├── scripts/
│   ├── training/
│   ├── vision_v2/
│   └── visualizations/
├── docs/
├── Dockerfile
├── docker-compose.yml
├── farm360.db
├── requirements.txt
├── README.md
└── TODO.md
```

---

## 2. Comprehensive Directory & Component Descriptions

### 2.1 `backend/` Subsystem Directories
- **`agent_core/`:** Contains response formatting and explainability components (`explainability.py`).
- **`api_gateway/`:** Houses model wrappers and LLM validation components (`model_wrapper.py`).
- **`core/`:** Core infrastructure components including database session management (`database.py`) and Fernet AES-256 PII security encryptor (`security.py`).
- **`decision_engine/`:** Rule-based expert system logic for agricultural decisions (`rule_engine.py`).
- **`evaluation/`:** Model evaluation scripts and benchmarks.
- **`external_apis/`:** Integration clients for third-party services like OpenWeather (`weather.py`).
- **`feedback/`:** User feedback logging and quality metrics store.
- **`media_pipeline/`:** Image processing and conversion utilities (`image_processor.py`).
- **`memory/`:** Legacy JSON conversation session serializer (`session.py`).
- **`memory_v2/`:** Advanced multi-tier memory architecture (`memory_store.py`, `memory_ranker.py`, `memory_retriever.py`, `memory_scheduler.py`, `memory_summarizer.py`).
- **`models/`:** SQLAlchemy ORM database models (`database.py`) defining 18 entities (`User`, `ChatSession`, `MemoryRecord`, `Prediction`, `DocumentChunk`).
- **`observability/`:** Async `EventBus`, `TraceContext`, and structured logging engine.
- **`performance/`:** Latency benchmarking and caching stubs.
- **`prompts/templates/`:** Jinja2 prompt templates (`general_assistant_v1.0.0.jinja2`, `vision_crop_disease_v1.0.0.jinja2`, etc.).
- **`rag/`:** Retrieval-Augmented Generation subsystem (`chunker.py`, `embedder.py`, `retriever.py`, `reranker.py`, `service.py`).
- **`repositories/`:** SQLAlchemy unit of work pattern and data access repositories (`memory_repo.py`).
- **`router/`:** Intent classification and domain advisor routing registry (`advisors.py`, `router.py`, `registry.py`).
- **`services/`:** Business domain services including health check service (`health_service.py`) and DB service (`database_service.py`).
- **`streaming/`:** SSE streaming engine (`stream_manager.py`, `stream_events.py`, `stream_response.py`, `stream_metrics.py`).
- **`tests/`:** Pytest unit and integration test suite.
- **`validator/`:** Safety and policy guardrail engine (`engine.py`) and rules (`rules/`).
- **`vision_service/`:** Multi-task computer vision service engine (`engine.py`, `registry.py`, `security.py`, `monitoring.py`) and REST routes (`routes/crop_disease.py`, `routes/breed.py`, etc.).

---

### 2.2 `frontend/` Web Client Directory
- **`src/app/`:** Next.js 16 App Router pages (`page.tsx`, `layout.tsx`, `globals.css`).
- **`src/components/`:** React 18 UI components (`ChatCanvas.tsx`, `ChatInput.tsx`, `Sidebar.tsx`, `VisionUpload.tsx`, `ErrorBoundary.tsx`).

---

### 2.3 `machine_learning/` Sub-Module Directories
- **`crop_regression/`:** scikit-learn crop yield prediction models and training scripts.
- **`crop_vision/`:** PyTorch ResNet18 17-class crop disease classifier weights and training scripts.
- **`dairy_module/`:** Time-series milk collection forecasting models.
- **`animal_module/` & `health_module/`:** Livestock classification and disease diagnosis models.
- **`models/`:** Serialized binary weight artifacts (`.pkl`, `.pth`).
- **`vision_v2/`:** Experimental next-generation vision pipelines.

---

### 2.4 `docs/` Documentation Directory
Contains 19 production-grade markdown files representing the complete, self-maintaining memory of the project.
