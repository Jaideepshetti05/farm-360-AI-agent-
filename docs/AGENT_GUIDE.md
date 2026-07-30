# Farm360 AI — Coding Agent Guide (`AGENT_GUIDE.md`)

> **Mandatory Guidelines & Engineering Protocols for AI Coding Agents**  
> **Version:** 3.1.0  
> **Last Updated:** 2026-07-30  

---

## 1. Mandatory Pre-Implementation Protocol

Every AI agent or developer working on this codebase **MUST** complete the following 6-step checklist before making any code modifications:

1. **Read `AGENT_GUIDE.md`** (this document) for engineering rules.
2. **Read `PROJECT_MEMORY.md`** to understand the global system architecture and components.
3. **Read `PROJECT_STATE.md`** to verify current implementation status and active blockers.
4. **Read `AI_CONTEXT.md`** to understand business goals, philosophy, and architectural constraints.
5. **Read `IMPLEMENTATION_RULES.md`** for coding standards and anti-duplication principles.
6. **Read `MODULE_INDEX.md`** to locate existing services, classes, and helper utilities.

---

## 2. Core Architecture Principles

1. **Never Duplicate Existing Functionality:** Always search the codebase using grep/find tools before creating new classes, helper functions, or services.
2. **Loosely Coupled Services:** Keep domain components (`memory_v2`, `vision_service`, `rag`, `provider_manager`, `validator`) strictly isolated behind explicit Python interfaces.
3. **Fail-Safe Fallbacks:** Never allow third-party API limits or database disconnects to crash the primary application. Always provide clean fallback logic.
4. **Zero Hardcoded Secrets:** Never commit API keys, passwords, or Fernet encryption keys to source control. Always read from environment variables with fallback warnings.

---

## 3. Code Analysis & Investigation Rules

- **Inspect authorative files:** Never guess API signatures, table schemas, or component prop names. Always read the exact definition file (`backend/models/database.py`, `backend/config.py`, etc.).
- **Read Error Tracebacks:** If a server command or unit test fails, your very first action must be to inspect the full error traceback before attempting a fix.
- **No Masking of Symptoms:** Fix root causes. Never resolve failures by swallowing exceptions silently or commenting out assertions.

---

## 4. Testing & Verification Standards

- **Backend Verification:** Execute unit tests using `pytest backend/tests/` after modifying backend modules.
- **Frontend Verification:** Ensure Next.js builds cleanly via `npm run build` inside `frontend/`.
- **API Contract Verification:** Test endpoints using `curl` or python test scripts (`test_llm.py`, `test_stream.py`) to confirm valid status codes and JSON payloads.

---

## 5. Post-Implementation Documentation Sync

After completing any code edit, the agent **MUST** update the affected documentation files in `docs/`:

- Always update `PROJECT_STATE.md` and `SESSION_SUMMARY.md`.
- Update `IMPLEMENTATION_LOG.md` with a detailed record of the change.
- Update `TODO.md` to check off completed tasks.
- If architectural changes occurred, update `ARCHITECTURE.md`, `PROJECT_MEMORY.md`, and `DECISIONS.md`.
- If new modules were created, update `MODULE_INDEX.md`.

---

## 6. Implementation Report Requirement

At the conclusion of every turn, future agents must provide a structured Implementation Report containing:
1. **Summary of Work Completed**
2. **Files Created & Modified**
3. **Tests & Commands Executed**
4. **Risks or Trade-offs Identified**
5. **Documentation Files Synchronized**
6. **Recommended Next Task**
