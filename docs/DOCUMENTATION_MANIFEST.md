# Farm360 AI — Documentation Manifest (`DOCUMENTATION_MANIFEST.md`)

> **Version:** 3.1.0  
> **Generated Date:** 2026-07-30  
> **Last Updated:** 2026-07-30  
> **Repository Commit:** cf67062  
> **Generator:** Antigravity AI Architecture Engine  
> **Status:** Active / Master Inventory  

---

## 1. Documentation Suite Catalog (19 Files)

The Farm360 AI project maintains a 19-file documentation suite inside the `docs/` directory. Each file serves a dedicated purpose and has explicit synchronization rules.

| Document Name | Purpose & Contents | Dependencies | Update Frequency | Owner | Required Update Trigger |
|---|---|---|---|---|---|
| **[AGENT_HANDOFF.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/AGENT_HANDOFF.md)** | Primary starting point for new sessions: active phase, branch, health, risks, next task, files, prompt. | `PROJECT_STATE.md` | Every Session | Lead Agent | Any code change / session end |
| **[PROJECT_STATE.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/PROJECT_STATE.md)** | Exact implementation status, module health matrix, production readiness score, and blockers. | `backend/`, `frontend/` | Every Session | Lead Architect | Code modifications or fixes |
| **[SESSION_SUMMARY.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/SESSION_SUMMARY.md)** | Active session summary, accomplished tasks, files needing developer focus, and handoff targets. | Current Trajectory | Every Session | Active Developer | End of work session |
| **[IMPLEMENTATION_LOG.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/IMPLEMENTATION_LOG.md)** | Historical Git commit record detailing modified files, rationale, and impact. | Git Log | Every Commit | Lead Architect | Major code commit |
| **[TODO.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/TODO.md)** | Categorized technical backlog (P0 Critical to P3 Low, Future Research). | Issues / Bugs | Every Sprint | Engineering Team | Task completed or added |
| **[CHANGELOG.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/CHANGELOG.md)** | Semantic version release history (v1.0.0 to v3.1.0). | Commit Log | Every Release | Release Lead | Version milestone tag |
| **[PROJECT_MEMORY.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/PROJECT_MEMORY.md)** | Master knowledge base: system vision, complete architecture, tech stack, DB schemas, APIs. | Entire Codebase | Major Release | Lead Architect | Architectural change |
| **[ARCHITECTURE.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/ARCHITECTURE.md)** | System diagrams (Mermaid) covering data flow, key failover, memory v2, vision, and streaming. | Backend Services | Architectural Shift | System Architect | Pipeline / Router change |
| **[DECISIONS.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/DECISIONS.md)** | Architectural Decision Records (ADRs 001-008) capturing technical choices and trade-offs. | System Design | On ADR Creation | Technical Lead | New technical decision |
| **[MODULE_INDEX.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/MODULE_INDEX.md)** | Complete directory index detailing responsibilities, public APIs, classes, and config per module. | Code Base | Module Change | Code Maintainer | New module or API added |
| **[CODEBASE_REPORT.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/CODEBASE_REPORT.md)** | Deep engineering audit, language stats, security/performance audits, and quality scores. | Code Audit | Quarterly | Security Auditor | Major audit / release |
| **[TRACEABILITY_MATRIX.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/TRACEABILITY_MATRIX.md)** | End-to-end feature-to-code mapping matrix (Feature → API → DB → ML → UI → Test → Doc). | Code Base | Major Release | QA / Architect | New feature implemented |
| **[REPOSITORY_STRUCTURE.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/REPOSITORY_STRUCTURE.md)** | Complete directory tree map explaining the purpose of every folder and sub-folder. | Directory Tree | Monthly | Maintainer | Directory restructuring |
| **[AI_CONTEXT.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/AI_CONTEXT.md)** | Permanent AI memory explaining vision, business goals, engineering philosophy, and rules. | Product Vision | Strategic Shift | Product Owner | Business goal change |
| **[AGENT_GUIDE.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/AGENT_GUIDE.md)** | Onboarding manual detailing coding standards, anti-duplication, and pre-implementation rules. | Project Policy | Bi-Annually | Lead Agent | Protocol modification |
| **[AGENT_PROMPT.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/AGENT_PROMPT.md)** | Turn-key reusable prompt template for initializing future AI coding sessions. | Agent Guide | As Needed | Prompt Engineer | Prompt enhancement |
| **[IMPLEMENTATION_RULES.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/IMPLEMENTATION_RULES.md)** | Permanent engineering rules, anti-duplication guidelines, doc sync policy, and 13-step workflow. | Project Policy | Bi-Annually | Lead Architect | Rule modification |
| **[MASTER_ROADMAP.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/MASTER_ROADMAP.md)** | Phased progress roadmap tracking completed, in-progress, and future features across 9 phases. | Product Backlog | Monthly | Product Manager | Phase completed |
| **[DOCUMENTATION_MANIFEST.md](file:///c:/Users/Jaideep/Desktop/ml%20models/docs/DOCUMENTATION_MANIFEST.md)** | Master inventory of all 19 documentation files, update frequencies, owners, and trigger rules. | Self-Referential | Major Release | Lead Architect | New doc file added |

---

## 2. Mandatory Synchronization Trigger Policy

After completing **ANY** code modification, future agents **MUST** execute documentation synchronization according to the matrix above.
