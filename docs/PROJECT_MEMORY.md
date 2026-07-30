# Farm360 AI — Master Project Memory (`PROJECT_MEMORY.md`)

> **Version:** 3.1.0  
> **Generated Date:** 2026-07-30  
> **Last Updated:** 2026-07-30  
> **Repository Commit:** cf67062  
> **Generator:** Antigravity AI Architecture Engine  
> **Status:** Active / Master Knowledge Base  

---

## 1. Executive Summary & Project Overview

**Farm360 AI** is a production-grade, multi-modal artificial intelligence platform designed for agricultural intelligence, pest diagnosis, yield prediction, and real-time decision support. It combines machine learning models (tabular regression, time-series forecasting, computer vision) with conversational large language models (LLMs) to provide real-time, actionable insights to farmers, agronomists, and agricultural enterprises.

### Core Capabilities
- **Crop Yield Forecasting:** Tabular regression predicting crop yields based on rainfall, fertilizer inputs, region, and land area.
- **Dairy Intelligence:** Time-series forecasting for milk collection volume, fat content, and SNF percentage.
- **Animal Health Diagnosis & Classification:** ML classifier for livestock condition diagnosis from symptoms and tag numbers.
- **Multi-Task Computer Vision (PyTorch ResNet18):** 17-class crop disease detection, livestock breed identification, weed classification, plant species identification, and fruit quality grading.
- **Multi-Provider LLM Orchestration:** Smart failover and key rotation across Google Gemini 2.5 Flash (native SDK), OpenRouter, and OpenAI key pools.
- **Memory v2 Engine:** Recency-, importance-, and vector-ranked memory store with PostgreSQL fallback to local JSON (`logs/memory_v2_fallback.json`).
- **Retrieval-Augmented Generation (RAG):** Agricultural knowledge chunking with 768-dimensional Gemini embeddings and re-ranking.
- **Real-Time Streaming Interface:** Server-Sent Events (SSE) streaming chat delivered to a Next.js 16 frontend.

---

## 2. Empirical Repository Statistics

| Metric | Measured Value | Scope / Description |
|---|---|---|
| **Total Clean Folders** | 438 | Directories (excluding `node_modules`, `.git`, `.venv`) |
| **Total Python Files** | 180 | Backend services, ML training scripts, tests |
| **Backend Core Files** | 139 | FastAPI server, routers, services, models |
| **Frontend Source Files** | 12 | Next.js 16 App Router TSX/TS components |
| **SQLAlchemy ORM Models** | 18 | Declarative database tables in `backend/models/database.py` |
| **Active REST Endpoints** | 14 | FastAPI REST and SSE endpoints (`app.py` & vision routes) |
| **Jinja2 Prompt Templates** | 7 | Versioned templates under `backend/prompts/templates/` |
| **Registered Vision Routers** | 6 | Task routers (`crop-disease`, `breed`, `weed`, `detect`, `plant-id`, `fruit-grade`) |
| **Machine Learning Datasets** | 94,869 files | Image dataset samples for crop vision training |
| **Clean Source Lines of Code** | ~13,732 LOC | Python (12,450 LOC) + TypeScript (1,282 LOC) |

---

## 3. Technology Stack Breakdown

| Layer | Technology / Library | Version / Purpose |
|---|---|---|
| **Backend Web Framework** | FastAPI + Uvicorn | v0.110+, Async web framework with Pydantic v2 validation |
| **Frontend Framework** | Next.js 16 (App Router), React 18 | SSR/CSR hybrid web application in TypeScript |
| **Styling Engine** | Tailwind CSS | Utility-first CSS styling system |
| **LLM SDKs** | `google-genai` (v1.x), `openai` (v1.x) | Native Gemini 2.5 Flash and OpenAI-compatible client |
| **ML Frameworks** | PyTorch, scikit-learn, torchvision | Computer vision & tabular regression models |
| **Image Processing** | Pillow (PIL), OpenCV | Image preprocessing, resizing, and normalization |
| **Database & ORM** | SQLAlchemy 2.0 (Async), PostgreSQL / SQLite | Declarative ORM with `pgvector` fallback |
| **PII Encryption** | Cryptography (Fernet AES-256) | Symmetrical field-level database encryption |
| **Prompt Engine** | Jinja2 | Templated prompt management system |
| **Streaming Engine** | Server-Sent Events (SSE) | HTTP line-delimited `data: [token]` streaming |
| **Observability** | Loguru | Asynchronous structured logging and event bus |
| **Containerization** | Docker, Docker Compose | Python 3.11 base image container orchestration |

---

## 4. Subsystem Architectures

### 4.1 Multi-Provider Key Rotation (`backend/provider_manager.py`)
- **Key Pools:** Ingests `GOOGLE_API_KEY_1...5`, `OPENROUTER_API_KEY_1...5`, `OPENAI_API_KEY_1...3`.
- **Rotation Rule:** On HTTP 429 / Rate limit, key is quarantined for 60 seconds. On HTTP 401 / Invalid Key, key is permanently disabled.
- **Failover Chain:** Gemini pool → OpenRouter pool → OpenAI pool → Deterministic Rule Engine.

### 4.2 Computer Vision Engine (`backend/vision_service/`)
- **Backbone:** PyTorch ResNet18 fine-tuned on 17 crop disease classes.
- **Explainability:** Grad-CAM heatmap generation integrated into model outputs.
- **Security:** `sanitize_filename()` generates random UUID filenames to protect against path traversal attacks.

### 4.3 Memory v2 Subsystem (`backend/memory_v2/`)
- **Scoring Model:** Hybrid linear combination:
  $$\text{Score} = w_1 \cdot \text{Recency} + w_2 \cdot \text{Importance} + w_3 \cdot \text{Frequency} + w_4 \cdot \text{Similarity}$$
- **Persistence:** Direct PostgreSQL writes via SQLAlchemy `UnitOfWork` with fallback to `logs/memory_v2_fallback.json`.

---

## 5. Master API Inventory

| Endpoint | Method | Inputs | Returns | Auth Required | Description |
|---|---|---|---|---|---|
| `/` | GET | None | JSON | No | Health check & model status |
| `/health/readiness` | GET | None | JSON | No | Service readiness probe |
| `/health/liveness` | GET | None | JSON | No | Liveness probe |
| `/chat_stream` | POST | `query`, `session_id`, `model` | SSE Stream | Yes (`X-API-Key`) | Real-time token streaming chat |
| `/chat` | POST | `query`, `model` | JSON | Yes (`X-API-Key`) | Blocking text chat endpoint |
| `/analyze_image` | POST | `query`, `model`, `image` | JSON | Yes (`X-API-Key`) | Multimodal image diagnosis |
| `/keys/status` | GET | None | JSON | Yes (`X-API-Key`) | Key pool health & cooldown status |
| `/vision/crop-disease` | POST | Image upload | JSON | Yes (`X-API-Key`) | ResNet18 crop disease classifier |
| `/vision/breed` | POST | Image upload | JSON | Yes (`X-API-Key`) | Animal breed classifier |
| `/vision/models` | GET | None | JSON | No | Registered vision model status |

---

## 6. Critical Security & Performance Notes

- **🔴 P0 Security Vulnerability:** `frontend/src/components/ChatInput.tsx` reads `NEXT_PUBLIC_FARM360_API_KEY`, exposing backend API secrets in browser bundles. (Fix: Implement Next.js API route proxy).
- **🟠 P1 Scaling Bottleneck:** `/chat_stream` spawns OS background threads per connection (`threading.Thread`). (Fix: Refactor to `asyncio.Queue`).
