# AI SRE Copilot

Hệ thống AI hỗ trợ Site Reliability Engineers (SRE) trong việc phân tích nguyên nhân gốc (Root Cause Analysis), tìm kiếm runbooks và tự động tổng hợp thông tin từ logs/metrics/traces.

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-4o-mini + Ollama (fallback) |
| Vector DB | Milvus 2.4 |
| Logs/Metrics/Traces | Loki / Prometheus / Tempo |
| LLM Observability | Langfuse |
| Cache | Redis |
| Gateway | FastAPI |
| Orchestration | Airflow |

## Quick Start

### 1. Cấu hình Environment

```bash
cp example.env .env
```

Chỉnh sửa `.env`:

```bash
# API Gateway
SRE_API_KEYS=local-key
SRE_RATE_LIMIT_PER_MINUTE=60

# LLM (chọn 1 hoặc cả 2)
SRE_OPENAI_API_KEY=sk-xxx
SRE_OLLAMA_BASE_URL=http://localhost:11434
SRE_OLLAMA_CHAT_MODEL=llama3

# Storage
SRE_MILVUS_URI=http://localhost:19530
SRE_REDIS_URL=redis://localhost:6379/0
```

### 2. Khởi động Infrastructure

```bash
cd ../infras
docker compose up -d
```

### 3. Cài đặt và chạy

```bash
# Install dependencies
uv sync

# Run API server
uv run uvicorn sre_copilot.app:app --port 8055 --reload
```

### 4. Test API

```bash
# Search runbooks
curl -X POST http://localhost:8055/v1/search \
  -H "Content-Type: application/json" \
  -H "x-api-key: local-key" \
  -d '{"query": "high latency in payment service"}'

# Get answer with RAG
curl -X POST http://localhost:8055/v1/answer \
  -H "Content-Type: application/json" \
  -H "x-api-key: local-key" \
  -d '{"query": "Why did latency spike?"}'
```

## Features

- **Gateway Layer**: API key authentication + Redis-based rate limiting
- **Guardrails**: NeMo Guardrails với PII detection và prompt injection prevention
- **LLM Routing**: OpenAI primary với Ollama fallback tự động
- **RAG Pipeline**: Load → Chunk → Embed → Milvus
- **Observability**: Langfuse tracing + OpenTelemetry (Tempo)
- **Caching**: Redis cache cho retrieval responses

## Data Ingestion

Ingest logs/metrics/traces vào Milvus:

```bash
# Manual run
uv run python -m sre_copilot.ingestion.pipeline

# Via Airflow (http://localhost:8080)
# Enable DAG: observability_ingest
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/search` | POST | Search runbooks với guardrails |
| `/v1/answer` | POST | RAG-based answer generation |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

## CI/CD

- **CI**: Lint (ruff) + Test (pytest) on PR
- **CD**: Build Docker → Push ghcr.io → Deploy via docker compose

```bash
# Build locally
docker build -t sre-copilot .

# Run with compose
docker compose up -d
```

## Project Structure

```
sre-copilot/
├── src/sre_copilot/
│   ├── app.py              # FastAPI application
│   ├── config.py           # Settings
│   ├── guardrails/         # NeMo + PII + Injection detection
│   ├── ingestion/          # Load-Chunk-Embed pipeline
│   ├── middleware/         # Auth + Rate limit
│   ├── observability/      # Langfuse + OpenTelemetry
│   ├── routers/            # API endpoints
│   └── services/           # LLM router, Milvus, Redis
├── airflow/dags/           # Airflow DAGs
├── tests/                  # Unit tests
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## License

MIT
