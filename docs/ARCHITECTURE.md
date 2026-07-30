# Farm360 AI — Architecture Blueprint (`ARCHITECTURE.md`)

> **Comprehensive Technical Architecture & Component Diagrams**  
> **Version:** 3.1.0  
> **Last Updated:** 2026-07-30  

---

## 1. System Overview

Farm360 AI is architected as a micro-service inspired, modular monolith built on FastAPI and Next.js 16. The backend encapsulates machine learning inference, LLM provider orchestration, memory retrieval, RAG, and computer vision services.

---

## 2. Master System Data Flow Diagram

```mermaid
flowchart TD
    User["Farmer / Web Client"] -- HTTP POST / SSE --> NextJS["Next.js 16 Frontend\n(ChatCanvas / VisionUpload)"]
    NextJS -- REST / EventSource --> APIGateway["FastAPI Core App\n(app.py)"]
    
    subgraph Security & Middleware
        RateLimiter["Sliding Window Rate Limiter"]
        APIKeyVerify["Constant-Time API Key Verifier"]
        FernetEnc["Fernet AES-256 PII Encryptor"]
    end

    APIGateway --> Security & Middleware
    Security & Middleware --> AgentCore["Farm360Agent Orchestrator\n(backend/main.py)"]

    subgraph Decision & Routing Layer
        IntentRouter["Intent Router\n(backend/router/router.py)"]
        Advisors["Domain Advisors\n(Crop, Livestock, Weather, General)"]
        ValidatorEngine["Validator Guardrails\n(backend/validator/engine.py)"]
    end

    AgentCore --> IntentRouter
    IntentRouter --> Advisors
    Advisors --> ValidatorEngine

    subgraph Memory & Context Layer
        MemoryStore["Memory v2 Store\n(backend/memory_v2/)"]
        MemoryRanker["Memory Ranker\n(Recency + Importance + Freq + Sim)"]
        RAGService["RAG Engine\n(backend/rag/service.py)"]
    end

    AgentCore --> MemoryStore
    MemoryStore --> MemoryRanker
    AgentCore --> RAGService

    subgraph ML & Vision Layer
        MLWrapper["FarmAPIWrapper\n(Crop, Dairy, Animal ML)"]
        VisionRegistry["Vision Model Registry\n(ResNet18 PyTorch Models)"]
    end

    AgentCore --> MLWrapper
    AgentCore --> VisionRegistry

    subgraph LLM Provider Layer
        ProviderManager["Provider Manager\n(backend/provider_manager.py)"]
        GeminiSDK["Google Gemini 2.5 Flash\n(Native google-genai SDK)"]
        OpenRouterSDK["OpenRouter Failover Pool\n(OpenAI SDK)"]
        OpenAISDK["OpenAI Fallback Pool\n(gpt-4o-mini)"]
    end

    AgentCore --> ProviderManager
    ProviderManager -->|Primary Pool| GeminiSDK
    ProviderManager -->|Secondary Failover| OpenRouterSDK
    ProviderManager -->|Tertiary Fallback| OpenAISDK

    subgraph Response Generation
        SSEStreamer["SSE Token Streamer\n(StreamingResponse)"]
    end

    ProviderManager --> SSEStreamer
    SSEStreamer -- data: token --> NextJS
```

---

## 3. Detailed Subsystem Architectures

### 3.1 Multi-Provider Key Rotation & Failover Architecture

```mermaid
sequenceDiagram
    autonumber
    participant Agent as Farm360Agent
    participant PM as ProviderManager
    participant PoolG as Gemini Key Pool (1..5)
    participant PoolOR as OpenRouter Pool (1..5)
    participant RuleEngine as Deterministic Rule Engine

    Agent->>PM: Request Stream Completion (query, prompt)
    PM->>PoolG: Select active Gemini key
    alt Gemini Key Healthy
        PoolG-->>PM: Yield stream tokens
        PM-->>Agent: Pass token stream
    else Gemini 429 Rate-Limited
        PoolG->>PoolG: Quarantined for 60s
        PM->>PoolG: Try next Gemini key slot
    else Gemini 401 Fatal Auth Error
        PoolG->>PoolG: Permanently disable key
        PM->>PoolOR: Failover to OpenRouter Key Pool
        PoolOR-->>PM: Yield stream tokens
        PM-->>Agent: Pass token stream
    else All LLM Keys Exhausted
        PM->>RuleEngine: Trigger rule-based fallback response
        RuleEngine-->>PM: Return deterministic advice
        PM-->>Agent: Yield character-by-character fallback
    end
```

---

### 3.2 Memory v2 Hybrid Ranking Architecture

```mermaid
flowchart LR
    subgraph Input Query & Candidates
        Query["User Query"]
        RawMemories["Stored Memory Records"]
    end

    subgraph Feature Extraction
        RecencyCalc["Recency Decay Calculation"]
        ImportanceCalc["Static Importance Weight"]
        FreqCalc["Access Frequency Score"]
        VectorSim["Cosine Similarity Score (768-dim)"]
    end

    subgraph Weighted Scoring Engine
        Ranker["MemoryRanker\nScore = w1·R + w2·I + w3·F + w4·S"]
    end

    subgraph Storage Tier
        DB["PostgreSQL DB (UnitOfWork)"]
        JSONFallback["JSON Storage (memory_v2_fallback.json)"]
    end

    Query --> VectorSim
    RawMemories --> RecencyCalc
    RawMemories --> ImportanceCalc
    RawMemories --> FreqCalc
    RawMemories --> VectorSim

    RecencyCalc --> Ranker
    ImportanceCalc --> Ranker
    FreqCalc --> Ranker
    VectorSim --> Ranker

    Ranker --> TopK["Top-K Ranked Memories"]
    TopK --> DB
    DB -. Connection Error .-> JSONFallback
```

---

### 3.3 Multi-Task Computer Vision Pipeline

```mermaid
flowchart TD
    Client["Image Upload (POST /analyze_image)"] --> Sanitize["Filename Sanitizer (UUID.hex)"]
    Sanitize --> TempDisk["Temp Disk Storage (temp_uploads/)"]
    TempDisk --> Preprocessor["Image Preprocessor\n(Resize 224x224, Normalize, ToTensor)"]
    
    Preprocessor --> TaskRouter{"Vision Task Router"}
    
    TaskRouter -->|Crop Disease| ResNetCrop["ResNet18 PyTorch\n(17 Crop Disease Classes)"]
    TaskRouter -->|Breed ID| ResNetBreed["Livestock Breed Classifier"]
    TaskRouter -->|Weed ID| ResNetWeed["Weed Infestation Model"]
    TaskRouter -->|Pest Detect| YoloDetect["Bounding Box Pest Detector"]
    
    ResNetCrop --> GradCAM["Grad-CAM Heatmap Explainer"]
    GradCAM --> Formatting["Schema Formatter (schemas.py)"]
    Formatting --> CleanDisk["Cleanup Temp File"]
    CleanDisk --> JSONResult["JSON Diagnosis Response"]
```

---

## 4. Architectural Principles

1. **Decoupled Business Logic:** All ML inference and vision tasks are isolated behind service interfaces.
2. **Fail-Safe Design:** Single key rate-limits or database connection failures will not crash the application.
3. **Zero Hardcoded Secrets:** All secrets load dynamically from `.env` or generate cryptographically secure temporary defaults with explicit logger warnings.
