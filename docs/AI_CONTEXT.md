# Farm360 AI — Strategic AI Context & Philosophy (`AI_CONTEXT.md`)

> **Permanent AI Memory, System Philosophy, & Non-Negotiable Engineering Directives**  
> **Version:** 3.1.0  
> **Last Updated:** 2026-07-30  

---

## 1. Project Purpose & Strategic Vision

**Farm360 AI** exists to democratize artificial intelligence for global agricultural producers. Smallholder farmers and commercial agricultural managers face unprecedented climate variability, pest pressures, and soil degradation. Farm360 AI serves as an autonomous, multi-modal agricultural co-pilot that synthesizes field imagery, sensor telemetry, market data, and expert agronomical knowledge to provide immediate, actionable guidance.

---

## 2. Strategic Philosophy & Core Principles

### 2.1 System & Engineering Philosophy
- **Modular Monolith First:** Maintain high cohesion and low coupling within a unified FastAPI process before breaking into micro-services.
- **Fail-Safe Autonomy:** The system must never present a hard failure to a farmer in the field. When external APIs or LLM providers fail, the system gracefully falls back to deterministic rule engines and local ML models.
- **Zero-Trust Data Protection:** Farmer location and crop data are confidential assets. All sensitive fields are encrypted at rest using AES-256 Fernet encryption.

### 2.2 Artificial Intelligence & LLM Philosophy
- **Multi-Provider Failover:** No single AI provider (Google, OpenAI, Anthropic) should create a vendor lock-in or single point of failure.
- **Grounding & Guardrails:** LLM outputs must be validated by domain guardrail rules (`backend/validator/`) to prevent dangerous agricultural recommendations (e.g., incorrect pesticide dosage).
- **Explainable Computer Vision:** Computer vision predictions should provide visual explanation artifacts (Grad-CAM heatmaps) to build farmer trust.

---

## 3. Non-Negotiable Directives ("Things Future Agents Must Never Change")

1. **NEVER Hardcode Credentials:** Secrets must be ingested via environment variables (`.env`). If an env var is missing, auto-generate a secure temporary token and log an explicit warning.
2. **NEVER Bypass Fallback Logic:** The `ProviderManager` must always attempt secondary and tertiary failover before yielding an error message.
3. **NEVER Remove Symmetrical PII Encryption:** Sensitive SQLAlchemy columns (`EncryptedString`) must remain encrypted at rest.
4. **NEVER Modify Business Logic Without Documentation Sync:** Any change to API endpoints, database models, or LLM providers must be immediately reflected across `docs/`.

---

## 4. Current Project Maturity Assessment

- **Core Engine:** Mature / Production-Ready (v3.1)
- **Multi-Provider Rotation:** Mature / Production-Ready
- **Computer Vision Service:** Production-Ready Core (17 crop disease classes active)
- **Memory v2 Engine:** Active Beta (PostgreSQL + JSON fallback operational)
- **RAG Subsystem:** Active Development (Chunker & Retriever active; batch ingestion pending)
