# Phân Tích Góc Độ Business Analyst (Technical)

**Dự án:** AI SRE Copilot
**Ngày phân tích:** 12/01/2026
**Đánh giá tổng thể:** 7.5/10

---

## 1. Executive Summary

Từ góc độ BA Technical, dự án có kiến trúc tốt với separation of concerns rõ ràng, async-first design, và comprehensive error handling. Tuy nhiên, còn một số gaps về integrations, API design, và data pipeline scalability.

---

## 2. System Overview

### 2.1 Application Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AI SRE COPILOT                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    PRESENTATION LAYER                          │ │
│  │  FastAPI Routers: /v1/search, /v1/answer, /health             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                      GATEWAY LAYER                             │ │
│  │  API Key Auth │ Rate Limiting │ Request Validation            │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    GUARDRAILS LAYER                            │ │
│  │  NeMo Guardrails │ PII Detection │ Prompt Injection Block     │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                     SERVICE LAYER                              │ │
│  │  Retrieval Service │ LLM Router │ Embedding Service           │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                      CACHE LAYER                               │ │
│  │  Redis Cache (TTL=300s) │ In-memory Fallback                  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                     STORAGE LAYER                              │ │
│  │  Milvus Vector DB │ ETcd (metadata) │ MinIO (objects)         │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                   OBSERVABILITY LAYER                          │ │
│  │  Langfuse │ OpenTelemetry │ Prometheus │ Loki │ Tempo         │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| API Framework | FastAPI | 0.115+ | Async REST API |
| Runtime | Uvicorn | Latest | ASGI server |
| Validation | Pydantic | 2.10+ | Request/Response models |
| LLM Primary | OpenAI | 1.55+ | GPT-4o-mini inference |
| LLM Fallback | Ollama | Latest | Local LLM inference |
| Guardrails | NeMo Guardrails | 0.10+ | Safety filtering |
| Vector DB | Milvus | 2.4.8 | Semantic search |
| Cache | Redis | 7-alpine | Response caching |
| Scheduler | Airflow | 2.9.3 | Data ingestion DAGs |
| Tracing | OpenTelemetry | 1.28+ | Distributed tracing |
| LLM Observability | Langfuse | 2.46+ | LLM call tracking |

---

## 3. Functional Requirements Analysis

### 3.1 Requirements Matrix

| Req ID | Requirement | Priority | Status | Test Coverage |
|--------|-------------|----------|--------|---------------|
| FR-001 | Semantic search for runbooks | P0 | ✅ Done | Yes |
| FR-002 | RAG-based answer generation | P0 | ✅ Done | Yes |
| FR-003 | API key authentication | P0 | ✅ Done | Yes |
| FR-004 | Rate limiting (60 req/min) | P0 | ✅ Done | Yes |
| FR-005 | PII detection and masking | P0 | ✅ Done | Yes |
| FR-006 | Prompt injection blocking | P0 | ✅ Done | Yes |
| FR-007 | Multi-source data ingestion | P1 | ✅ Done | Partial |
| FR-008 | Response caching | P1 | ✅ Done | Yes |
| FR-009 | LLM fallback (OpenAI→Ollama) | P1 | ✅ Done | Yes |
| FR-010 | Health check endpoints | P1 | ✅ Done | Yes |
| FR-011 | Batch search API | P2 | ❌ Missing | - |
| FR-012 | Streaming responses | P2 | ❌ Missing | - |
| FR-013 | Query rewriting | P2 | ✅ Done | No |
| FR-014 | Custom runbook upload | P1 | ❌ Missing | - |
| FR-015 | User management | P1 | ❌ Missing | - |

### 3.2 API Specification

#### 3.2.1 Current Endpoints

| Endpoint | Method | Auth | Request | Response |
|----------|--------|------|---------|----------|
| `/v1/search` | POST | Yes | `QueryRequest` | `SearchResponse` |
| `/v1/answer` | POST | Yes | `QueryRequest` | `AnswerResponse` |
| `/health` | GET | No | - | `HealthResponse` |
| `/health/ready` | GET | No | - | `ReadyResponse` |
| `/docs` | GET | No | - | Swagger UI |

#### 3.2.2 Request/Response Models

```python
# Request Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=3, ge=1, le=20)

# Response Models
class SearchResult(BaseModel):
    id: str
    text: str
    score: float

class SearchResponse(BaseModel):
    query: str
    sanitized_query: str
    results: List[SearchResult]
    guardrail_triggered: bool
    cached: bool

class AnswerResponse(SearchResponse):
    answer: str
    model: str
```

#### 3.2.3 Missing API Features

| Feature | Impact | Priority |
|---------|--------|----------|
| Batch search endpoint | Bulk operations | P2 |
| Streaming responses | Better UX for long answers | P1 |
| Pagination | Large result sets | P2 |
| Filtering by source | Targeted search | P2 |
| Query history | User convenience | P3 |

---

## 4. Non-Functional Requirements Analysis

### 4.1 Performance Requirements

| Requirement | Target | Current | Status |
|-------------|--------|---------|--------|
| API Latency (P50) | <500ms | ~200ms (cached) | ✅ Met |
| API Latency (P95) | <2s | ~3s (with LLM) | ⚠️ Partial |
| API Latency (P99) | <5s | Unknown | ❌ Not measured |
| Throughput | 100 req/s | ~60 req/s | ⚠️ Partial |
| Concurrent Users | 50 | Unknown | ❌ Not tested |

### 4.2 Scalability Requirements

| Requirement | Target | Current | Status |
|-------------|--------|---------|--------|
| Vector Storage | 1M vectors | ~500K max | ⚠️ Single-node limit |
| Horizontal Scaling | 10 nodes | 1 node | ❌ Not implemented |
| Auto-scaling | Yes | No | ❌ Not implemented |
| Multi-region | Yes | No | ❌ Not implemented |

### 4.3 Reliability Requirements

| Requirement | Target | Current | Status |
|-------------|--------|---------|--------|
| Availability | 99.9% | Unknown | ⚠️ No SLA |
| Failover Time | <30s | N/A | ❌ No HA |
| Data Durability | 99.99% | Unknown | ⚠️ Single node |
| Backup/Recovery | Daily | None | ❌ Not implemented |

---

## 5. Integration Analysis

### 5.1 Inbound Integrations (Data Sources)

| Source | Protocol | Status | Data Volume |
|--------|----------|--------|-------------|
| Grafana Loki | HTTP/LogQL | ✅ Working | Variable |
| Prometheus | HTTP/PromQL | ✅ Working | Variable |
| Grafana Tempo | HTTP API | ✅ Working | Variable |
| Confluence | ❌ Missing | Planned | High value |
| GitHub/GitLab | ❌ Missing | Planned | High value |
| Jira | ❌ Missing | Planned | Medium value |
| Custom Files | ❌ Missing | Planned | Variable |

### 5.2 Outbound Integrations (LLM Providers)

| Provider | Status | Fallback Order | Notes |
|----------|--------|----------------|-------|
| OpenAI | ✅ Primary | 1 | GPT-4o-mini |
| Ollama | ✅ Fallback | 2 | Local models |
| Azure OpenAI | ❌ Missing | Planned | Enterprise |
| AWS Bedrock | ❌ Missing | Planned | Enterprise |
| Anthropic Claude | ❌ Missing | Planned | Alternative |

### 5.3 Notification Integrations

| Platform | Status | Use Case |
|----------|--------|----------|
| Slack | ❌ Missing | Query bot |
| Microsoft Teams | ❌ Missing | Query bot |
| PagerDuty | ❌ Missing | Incident context |
| OpsGenie | ❌ Missing | Incident context |
| Email | ❌ Missing | Reports |
| Webhook | ❌ Missing | Custom integrations |

### 5.4 Integration Architecture (Recommended)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      INTEGRATION HUB                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │ Slack Bot   │  │ Teams Bot   │  │ PagerDuty   │                │
│  │ Adapter     │  │ Adapter     │  │ Adapter     │                │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │
│         │                │                │                        │
│         └────────────────┼────────────────┘                        │
│                          ▼                                          │
│              ┌─────────────────────┐                               │
│              │   Message Queue     │                               │
│              │   (Redis/RabbitMQ)  │                               │
│              └──────────┬──────────┘                               │
│                         │                                           │
│                         ▼                                           │
│              ┌─────────────────────┐                               │
│              │   API Gateway       │                               │
│              │   (Rate Limit, Auth)│                               │
│              └──────────┬──────────┘                               │
│                         │                                           │
│                         ▼                                           │
│              ┌─────────────────────┐                               │
│              │   SRE Copilot API   │                               │
│              └─────────────────────┘                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Flow Analysis

### 6.1 Query Processing Flow

```
                           ┌─────────────────┐
                           │   User Query    │
                           │ "Why is API     │
                           │  latency high?" │
                           └────────┬────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: GATEWAY                                                     │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│ │ API Key     │──│ Rate Limit  │──│ Validate    │                 │
│ │ Check       │  │ Check       │  │ Request     │                 │
│ └─────────────┘  └─────────────┘  └─────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: GUARDRAILS                                                  │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│ │ Prompt      │──│ PII         │──│ NeMo        │                 │
│ │ Injection   │  │ Detection   │  │ Guardrails  │                 │
│ └─────────────┘  └─────────────┘  └─────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: CACHE CHECK                                                 │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ Cache Key = "retrieval:k{top_k}:{sha1(query)}"                  ││
│ │ If HIT → Return cached response                                  ││
│ │ If MISS → Continue to Step 4                                     ││
│ └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: EMBEDDING                                                   │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ Primary: OpenAI text-embedding-3-small (dim=1536)               ││
│ │ Fallback 1: Ollama embeddings                                    ││
│ │ Fallback 2: SHA256 hash-based deterministic vector              ││
│ └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: VECTOR SEARCH                                               │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ Milvus Search:                                                   ││
│ │ - Collection: "runbooks"                                         ││
│ │ - Index: IVF_FLAT                                                ││
│ │ - Metric: Inner Product (IP)                                     ││
│ │ - Return: top_k results                                          ││
│ └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: LLM GENERATION (for /v1/answer only)                        │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ Prompt: "Based on the following context, answer the question:   ││
│ │         Context: {retrieved_chunks}                              ││
│ │         Question: {user_query}"                                  ││
│ │                                                                  ││
│ │ Primary: OpenAI GPT-4o-mini                                      ││
│ │ Fallback: Ollama local model                                     ││
│ └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 7: CACHE & RETURN                                              │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ - Store in Redis (TTL=300s)                                      ││
│ │ - Log to Langfuse                                                ││
│ │ - Export OTEL span                                               ││
│ │ - Return response to user                                        ││
│ └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Data Ingestion Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AIRFLOW DAG: observability_ingest                │
│                    Schedule: Every 2 minutes                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │ Fetch Loki  │  │ Fetch Prom  │  │ Fetch Tempo │                │
│  │ Logs        │  │ Metrics     │  │ Traces      │                │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │
│         │                │                │                        │
│         └────────────────┼────────────────┘                        │
│                          ▼                                          │
│              ┌─────────────────────┐                               │
│              │   Text Chunking     │                               │
│              │   (max 500 chars)   │                               │
│              └──────────┬──────────┘                               │
│                         │                                           │
│                         ▼                                           │
│              ┌─────────────────────┐                               │
│              │   Embedding         │                               │
│              │   (OpenAI/Ollama)   │                               │
│              └──────────┬──────────┘                               │
│                         │                                           │
│                         ▼                                           │
│              ┌─────────────────────┐                               │
│              │   Milvus Upsert     │                               │
│              │   (deduplicate)     │                               │
│              └─────────────────────┘                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Data Model

### 7.1 Milvus Collection Schema

```python
# Collection: runbooks
{
    "fields": [
        {
            "name": "id",
            "dtype": DataType.VARCHAR,
            "max_length": 256,
            "is_primary": True
        },
        {
            "name": "text",
            "dtype": DataType.VARCHAR,
            "max_length": 65535
        },
        {
            "name": "vector",
            "dtype": DataType.FLOAT_VECTOR,
            "dim": 1536  # OpenAI embedding dimension
        }
    ],
    "index": {
        "field_name": "vector",
        "index_type": "IVF_FLAT",
        "metric_type": "IP",  # Inner Product
        "params": {"nlist": 128}
    }
}
```

### 7.2 Redis Cache Schema

```python
# Cache Key Pattern
key = f"retrieval:k{top_k}:{sha1(query)}"

# Cache Value (JSON)
{
    "query": "original query",
    "sanitized_query": "cleaned query",
    "results": [
        {"id": "doc1", "text": "...", "score": 0.95},
        {"id": "doc2", "text": "...", "score": 0.87}
    ],
    "guardrail_triggered": false,
    "cached": true
}

# TTL: 300 seconds (5 minutes)
```

### 7.3 Missing Data Models

| Model | Purpose | Priority |
|-------|---------|----------|
| User | User management | P1 |
| Team | Team-based access | P1 |
| APIKey | Key management | P1 |
| QueryHistory | User query logs | P2 |
| Feedback | Answer ratings | P2 |
| Runbook | Custom runbooks | P1 |

---

## 8. Ưu Điểm (BA Tech View)

| # | Ưu điểm | Technical Impact |
|---|---------|------------------|
| 1 | **Clean layer separation** | Easy to maintain and extend |
| 2 | **Async-first design** | High concurrency support |
| 3 | **Comprehensive fallback chains** | High availability |
| 4 | **Type hints throughout** | Better IDE support, fewer bugs |
| 5 | **Pydantic validation** | Strong request/response contracts |
| 6 | **OpenTelemetry integration** | Production-grade observability |
| 7 | **Dependency injection** | Testable, mockable code |
| 8 | **Graceful degradation** | Resilient to failures |

---

## 9. Nhược Điểm (BA Tech View)

| # | Nhược điểm | Technical Impact | Priority |
|---|-----------|------------------|----------|
| 1 | **Limited data sources** | Only 3 sources | P1 |
| 2 | **No batch API** | Inefficient bulk ops | P2 |
| 3 | **No streaming responses** | Poor UX for long answers | P1 |
| 4 | **Sync Milvus client** | Blocking I/O | P2 |
| 5 | **Hardcoded chunk size** | Inflexible | P3 |
| 6 | **No pagination** | Large result issues | P2 |
| 7 | **Missing API versioning** | Breaking changes risk | P2 |
| 8 | **No webhook support** | Limited integrations | P2 |

---

## 10. Technical Debt

| Item | Location | Severity | Effort |
|------|----------|----------|--------|
| Sync Milvus calls | milvus_client.py | Medium | M |
| Hardcoded chunk size | pipeline.py:L45 | Low | S |
| No API versioning | routers/*.py | Medium | M |
| Missing input length limit | schemas/retrieval.py | High | S |
| No circuit breaker | llm_router.py | Medium | M |
| Duplicate code in tests | tests/*.py | Low | S |

---

## 11. Recommendations

### 11.1 Immediate (P0)
1. Add input length validation (prevent DoS)
2. Implement streaming for LLM responses
3. Add more data source connectors

### 11.2 Short-term (P1)
4. Implement batch search API
5. Add pagination support
6. Create webhook integration framework

### 11.3 Medium-term (P2)
7. Migrate to async Milvus client
8. Implement proper circuit breakers
9. Add API versioning (v2)

---

*Phân tích bởi: AI Expert Analysis*
*Phiên bản: 1.0*
