# Step 3.4.5 — Production Docker Compose + Caddy

## 1. Production Architecture
The production architecture is designed for single-node deployment on an AWS EC2 instance using Docker Compose and Caddy as an automated TLS reverse proxy:

```
                    Internet (Client / Browser)
                               │
                           80 / 443
                               │
                       ┌───────────────┐
                       │  Caddy (TLS)  │
                       └───────┬───────┘
                               │
                       ┌───────┴───────┐
                       │    Next.js    │ (Port 3000, private)
                       │   Frontend    │
                       └───────┬───────┘
                               │
                       ┌───────┴───────┐
                       │    FastAPI    │ (Port 8000, private)
                       │    Backend    │
                       └───────┬───────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
          ┌─────────┴─────────┐ ┌─────────┴─────────┐
          │    PostgreSQL     │ │       Redis       │
          │    + pgvector     │ │     L2 Cache      │
          │ (Port 5432, priv) │ │ (Port 6379, priv) │
          └───────────────────┘ └───────────────────┘
```

---

## 2. Service Topology
All services are orchestrated via [`docker-compose.prod.yml`](file:///c:/Users/Jaideep/Desktop/ml%20models/docker-compose.prod.yml):

| Service | Image / Build Context | Internal Port | Host Port Exposed | Role |
|---|---|---|---|---|
| **`caddy`** | `caddy:2-alpine` | 80, 443 | `80:80`, `443:443`, `443:443/udp` | Ingress gateway, TLS termination, HTTP➔HTTPS redirect, SSE streaming |
| **`frontend`** | `./frontend` (`Dockerfile`, standalone) | 3000 | *None* | SSR, UI static delivery, server-side API proxy |
| **`backend`** | `.` (`Dockerfile`, Python 3.11) | 8000 | *None* | FastAPI AI Agent, ML inference pipelines, Alembic migrations |
| **`postgres`** | `pgvector/pgvector:pg16` | 5432 | *None* | Relational DB & vector store (384-dim semantic embeddings) |
| **`redis`** | `redis:7-alpine` | 6379 | *None* | L2 cache, session state, rate limit records |

---

## 3. Network Exposure
- **Isolated Bridge Network:** All 5 services communicate over a private bridge network: `farm360-prod-net`.
- **Public Surface Area:** ONLY Caddy binds host ports `80` (HTTP) and `443` (HTTPS & HTTP/3 UDP).
- **Private Internal Ports:** Ports `3000`, `8000`, `5432`, and `6379` are strictly unmapped from the host network interface. No database or application service is directly accessible from the public internet.

---

## 4. Volumes
Persistent state is isolated in named Docker volumes:

| Volume Name | Target Path | Purpose |
|---|---|---|
| `postgres_data` | `/var/lib/postgresql/data` | PostgreSQL database files, tablespaces, and pgvector indices |
| `redis_data` | `/data` | Redis AOF (Append-Only File) persistence logs |
| `caddy_data` | `/data` | Let's Encrypt / ZeroSSL TLS certificates and private keys |
| `caddy_config` | `/config` | Caddy active runtime configuration state |

---

## 5. Environment Variables
Production configuration is externalized in [`.env.production.example`](file:///c:/Users/Jaideep/Desktop/ml%20models/.env.production.example). No plaintext secrets are committed:

- `DOMAIN`: Target FQDN (e.g. `app.farm360.ai`) or `:80` for initial EC2 Public IP smoke testing.
- `FARM360_API_KEY`: Mandatory shared secret for Next.js ➔ FastAPI proxy authentication.
- `FARM360_ENCRYPTION_KEY`: Mandatory 32-byte Fernet key for database PII encryption (`is_production()` fail-fast).
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`: Credentials injected into `postgres` and `backend`.
- `DATABASE_URL`: `postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}`.
- `REDIS_PASSWORD`: Optional Redis auth password.
- `CORS_ORIGINS`: Explicit allowed origin list (e.g. `https://app.farm360.ai`). Wildcard `*` strictly forbidden in production.
- `GOOGLE_API_KEY_1...5`, `OPENROUTER_API_KEY_1...5`, `OPENAI_API_KEY_1...3`: LLM API key pools.

---

## 6. Caddy Configuration
[`Caddyfile`](file:///c:/Users/Jaideep/Desktop/ml%20models/Caddyfile) implements:
1. **Dynamic Domain Binding:** `{$DOMAIN::80}` automatically toggles between HTTP `:80` (for IP tests) and full automatic HTTPS (when an FQDN is provided).
2. **Reverse Proxy:** Directs user traffic to `frontend:3000`.
3. **External Monitoring Passthrough:** Directs `/health*` directly to `backend:8000` for infrastructure/AWS health checks.
4. **Security Headers:** Enforces `X-Content-Type-Options "nosniff"`, `X-Frame-Options "DENY"`, `Referrer-Policy "strict-origin-when-cross-origin"`, and strips `-Server`.
5. **Compression:** High-performance `gzip` and `zstd` encoding.

---

## 7. HTTPS Requirements
- **Automated TLS:** When `DOMAIN` is set to an FQDN (e.g., `farm360.ai`), Caddy automatically requests and renews valid TLS certificates via Let's Encrypt / ZeroSSL using ACME HTTP-01 / TLS-ALPN-01 challenges on ports 80/443.
- **Initial IP Testing:** Before DNS is configured, setting `DOMAIN=:80` allows direct verification over HTTP via `http://<EC2_PUBLIC_IP>`.
- **AWS Security Group Requirement:** Inbound rules must allow TCP `80`, TCP `443`, and UDP `443` (for HTTP/3).

---

## 8. SSE / Streaming Configuration
Farm360 delivers conversational AI responses via Server-Sent Events (`/api/chat-stream`). To prevent buffering delays:
- In [`Caddyfile`](file:///c:/Users/Jaideep/Desktop/ml%20models/Caddyfile):
  ```caddy
  reverse_proxy frontend:3000 {
      flush_interval -1
  }
  ```
- `flush_interval -1` forces immediate flushing of each SSE chunk from upstream to downstream without buffering.
- Next.js server-side proxy maintains `X-Accel-Buffering: no`, `Cache-Control: no-cache, no-transform`, and `Connection: keep-alive`.

---

## 9. Security Validation
- [x] No secrets or credentials hardcoded in [`docker-compose.prod.yml`](file:///c:/Users/Jaideep/Desktop/ml%20models/docker-compose.prod.yml).
- [x] No secrets or credentials hardcoded in [`Caddyfile`](file:///c:/Users/Jaideep/Desktop/ml%20models/Caddyfile).
- [x] Ports 3000, 8000, 5432, 6379 are unmapped from the host network.
- [x] Only ports 80 and 443 are exposed.
- [x] Internal communication uses Docker DNS names (`postgres`, `redis`, `backend`, `frontend`).
- [x] `ENVIRONMENT=production` enforced on the backend container.
- [x] Non-root execution in frontend (`USER nextjs`).

---

## 10. Docker Validation
- **Compose Config Check:**
  ```powershell
  docker compose -f docker-compose.prod.yml config
  ```
  Result: **PASSED** (Exit code 0, all services, volumes, networks, and environment substitutions validated).
- **Service Verification:**
  ```powershell
  docker compose -f docker-compose.prod.yml config --services
  ```
  Result:
  ```
  redis
  postgres
  backend
  frontend
  caddy
  ```
- **Runtime Stack Execution:** **PENDING** on local host due to Docker Desktop daemon being stopped. Full container execution is ready for AWS EC2 instance launch.

---

## 11. AWS Deployment Prerequisites
Before running `docker compose -f docker-compose.prod.yml up -d` on AWS EC2:
1. **EC2 Instance:** Ubuntu 24.04 LTS (recommended: `t3.medium` or `t3.large`, 2 vCPU, 4GB RAM minimum for ML model loading).
2. **Security Group:**
   - Inbound: Port 22 (SSH), Port 80 (HTTP), Port 443 (HTTPS TCP/UDP).
3. **Docker Installation:** Docker Engine + Docker Compose Plugin (`apt install docker-compose-plugin`).
4. **Environment File:** Copy `.env.production.example` to `.env` on EC2, populate real secret keys (`FARM360_API_KEY`, `FARM360_ENCRYPTION_KEY`, `POSTGRES_PASSWORD`, `DOMAIN`, LLM keys).

---

## 12. Known Limitations
- Single-instance EC2 architecture does not include multi-AZ failover (sufficient for MVP deployment).
- Let's Encrypt requires a public domain with an A record pointing to the EC2 elastic IP; if accessing via raw IP, HTTP on port 80 must be used (`DOMAIN=:80`).
