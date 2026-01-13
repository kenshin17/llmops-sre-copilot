# Phân Tích Góc Độ An Toàn Thông Tin (ATTT)

**Dự án:** AI SRE Copilot
**Ngày phân tích:** 12/01/2026
**Đánh giá tổng thể:** 6/10
**Mức độ rủi ro:** TRUNG BÌNH - CAO

---

## 1. Executive Summary

Từ góc độ An toàn thông tin, dự án có một số controls tốt (guardrails, PII detection, rate limiting) nhưng thiếu nhiều yếu tố bảo mật quan trọng cho production: encryption at rest, RBAC, secrets management, và nhiều lỗ hổng cấu hình.

**Findings Summary:**
- 🔴 Critical: 3
- 🟠 High: 5
- 🟡 Medium: 7
- 🟢 Low: 4

---

## 2. Security Assessment Framework

### 2.1 Assessment Methodology

Đánh giá dựa trên:
- OWASP Top 10 (Web Application)
- OWASP Top 10 for LLM Applications
- CIS Benchmarks for Docker
- NIST Cybersecurity Framework

### 2.2 Scope

| Component | In Scope | Notes |
|-----------|----------|-------|
| API Application | ✅ Yes | FastAPI app |
| Infrastructure | ✅ Yes | Docker Compose |
| LLM Integration | ✅ Yes | OpenAI, Ollama |
| Data Storage | ✅ Yes | Milvus, Redis |
| Observability | ✅ Yes | Langfuse, OTEL |
| CI/CD | ⚠️ Partial | GitHub Actions |
| Cloud Provider | ❌ No | Not deployed yet |

---

## 3. Authentication & Authorization

### 3.1 Current Implementation

```
┌─────────────────────────────────────────────────────────────────────┐
│                 AUTHENTICATION FLOW                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Client Request                                                     │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ API Key Check (middleware/auth.py)                           │   │
│  │                                                              │   │
│  │ Header: x-api-key                                            │   │
│  │ Valid Keys: SRE_API_KEYS (comma-separated or JSON array)    │   │
│  │                                                              │   │
│  │ Results:                                                     │   │
│  │ - Missing key → 401 Unauthorized                            │   │
│  │ - Invalid key → 403 Forbidden                               │   │
│  │ - Valid key → Continue                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ⚠️ ISSUES:                                                        │
│  - No key expiration                                               │
│  - No key rotation mechanism                                        │
│  - No user identity (just valid/invalid)                           │
│  - No rate limit per key (only per IP)                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Authentication Findings

| Finding | Severity | Status |
|---------|----------|--------|
| API key authentication implemented | Info | ✅ Good |
| No key expiration/TTL | 🟠 High | ❌ Missing |
| No key rotation mechanism | 🟠 High | ❌ Missing |
| No OAuth2/OIDC support | 🟡 Medium | ❌ Missing |
| No MFA/2FA | 🟡 Medium | ❌ Missing |
| No brute force protection | 🟡 Medium | ⚠️ Partial |

### 3.3 Authorization Findings

| Finding | Severity | Status |
|---------|----------|--------|
| No RBAC implementation | 🟠 High | ❌ Missing |
| No resource-level permissions | 🟡 Medium | ❌ Missing |
| No team/tenant isolation | 🟡 Medium | ❌ Missing |
| All authenticated users have same access | 🟡 Medium | ⚠️ Risk |

---

## 4. Input Validation & Injection Prevention

### 4.1 Guardrails Implementation

```python
# Location: src/sre_copilot/guardrails/

┌─────────────────────────────────────────────────────────────────────┐
│                    GUARDRAILS CHAIN                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input Query                                                        │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 1. PROMPT INJECTION DETECTION (injection.py)                │   │
│  │                                                              │   │
│  │ Patterns detected:                                           │   │
│  │ - "ignore previous instructions"                             │   │
│  │ - "disregard above"                                          │   │
│  │ - "system prompt"                                            │   │
│  │ - "you are now"                                              │   │
│  │ - "drop all rules"                                           │   │
│  │ - "delete your instructions"                                 │   │
│  │ - "sudo rm"                                                  │   │
│  │                                                              │   │
│  │ ✅ Good coverage for common attacks                          │   │
│  │ ⚠️ Can be bypassed with encoding/obfuscation                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 2. PII DETECTION (pii.py)                                    │   │
│  │                                                              │   │
│  │ Detected patterns:                                           │   │
│  │ - Email addresses: user@example.com                          │   │
│  │ - API Keys: sk-xxx, pk_xxx, rk_xxx                          │   │
│  │ - IP Addresses: 192.168.1.1                                  │   │
│  │ - Phone Numbers: +1-234-567-8900                            │   │
│  │                                                              │   │
│  │ Action: Replace with [REDACTED]                              │   │
│  │                                                              │   │
│  │ ✅ Good for US/international formats                         │   │
│  │ ❌ Missing Vietnamese PII (CMND, CCCD, phone)               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 3. NEMO GUARDRAILS (engine.py)                               │   │
│  │                                                              │   │
│  │ Primary: NVIDIA NeMo Guardrails                              │   │
│  │ Fallback: Heuristic patterns (if NeMo unavailable)          │   │
│  │                                                              │   │
│  │ ✅ Dual-mode for reliability                                 │   │
│  │ ⚠️ Depends on OpenAI API key for NeMo                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Input Validation Findings

| Finding | Severity | Status |
|---------|----------|--------|
| Prompt injection detection | Info | ✅ Good |
| PII detection and masking | Info | ✅ Good |
| No input length limit | 🟠 High | ❌ Missing |
| No query rate per user | 🟡 Medium | ❌ Missing |
| Regex bypass possible | 🟡 Medium | ⚠️ Risk |
| No output sanitization | 🟡 Medium | ❌ Missing |

---

## 5. OWASP Top 10 Assessment

### 5.1 Web Application Security

| # | Vulnerability | Status | Evidence | Risk |
|---|---------------|--------|----------|------|
| A01 | Broken Access Control | ⚠️ | No RBAC, API key only | 🟠 High |
| A02 | Cryptographic Failures | ❌ | No encryption at rest | 🔴 Critical |
| A03 | Injection | ✅ | Guardrails mitigate | 🟢 Low |
| A04 | Insecure Design | ⚠️ | No threat model docs | 🟡 Medium |
| A05 | Security Misconfiguration | ❌ | Default creds in docker | 🔴 Critical |
| A06 | Vulnerable Components | ⚠️ | No dependency scanning | 🟡 Medium |
| A07 | Auth Failures | ⚠️ | No key rotation | 🟠 High |
| A08 | Data Integrity Failures | ⚠️ | No input signing | 🟡 Medium |
| A09 | Logging Failures | ✅ | Langfuse + OTEL | 🟢 Low |
| A10 | SSRF | ⚠️ | User queries to APIs | 🟡 Medium |

### 5.2 LLM Application Security (OWASP LLM Top 10)

| # | Risk | Status | Mitigation | Risk Level |
|---|------|--------|------------|------------|
| LLM01 | Prompt Injection | ✅ | NeMo + regex patterns | 🟡 Medium |
| LLM02 | Insecure Output | ⚠️ | No output sanitization | 🟡 Medium |
| LLM03 | Training Data Poisoning | ⚠️ | No data validation | 🟡 Medium |
| LLM04 | Model DoS | ✅ | Rate limiting | 🟢 Low |
| LLM05 | Supply Chain | ⚠️ | External LLM dependency | 🟡 Medium |
| LLM06 | Sensitive Info Disclosure | ✅ | PII masking | 🟢 Low |
| LLM07 | Insecure Plugin Design | N/A | No plugins | N/A |
| LLM08 | Excessive Agency | ✅ | Read-only operations | 🟢 Low |
| LLM09 | Overreliance | ⚠️ | No confidence scoring | 🟡 Medium |
| LLM10 | Model Theft | N/A | Using external APIs | N/A |

---

## 6. Data Security

### 6.1 Data Classification

| Data Type | Classification | Current Protection | Required |
|-----------|----------------|-------------------|----------|
| User Queries | Confidential | ⚠️ Logged, not encrypted | Encryption |
| API Keys | Secret | ⚠️ .env file | Vault |
| LLM Responses | Internal | ⚠️ Cached, not encrypted | Encryption |
| Runbook Content | Internal | ⚠️ Stored, not encrypted | Encryption |
| Observability Data | Internal | ✅ Grafana stack | OK |
| LLM API Keys | Secret | ⚠️ .env file | Vault |

### 6.2 Data Protection Findings

| Finding | Severity | Location | Status |
|---------|----------|----------|--------|
| No encryption at rest | 🔴 Critical | All storage | ❌ |
| No TLS for internal traffic | 🟠 High | Service-to-service | ❌ |
| Secrets in .env files | 🟠 High | Configuration | ❌ |
| No data retention policy | 🟡 Medium | All data | ❌ |
| No backup encryption | 🟡 Medium | Volumes | ❌ |
| PII in logs possible | 🟡 Medium | Langfuse | ⚠️ |

---

## 7. Infrastructure Security

### 7.1 Docker Security Assessment

```yaml
# File: infras/docker-compose.yml

CRITICAL FINDINGS:

1. Default Credentials (Line 32-33):
   ┌─────────────────────────────────────────────────┐
   │ MINIO_ACCESS_KEY: minioadmin                    │
   │ MINIO_SECRET_KEY: minioadmin                    │
   └─────────────────────────────────────────────────┘
   Risk: 🔴 CRITICAL - Hardcoded default credentials

2. Default Credentials (Line 119-120):
   ┌─────────────────────────────────────────────────┐
   │ GF_SECURITY_ADMIN_USER: admin                   │
   │ GF_SECURITY_ADMIN_PASSWORD: admin               │
   └─────────────────────────────────────────────────┘
   Risk: 🔴 CRITICAL - Hardcoded admin credentials

3. Default Credentials (Line 139-141):
   ┌─────────────────────────────────────────────────┐
   │ POSTGRES_USER: langfuse                         │
   │ POSTGRES_PASSWORD: langfuse                     │
   └─────────────────────────────────────────────────┘
   Risk: 🟠 HIGH - Weak database credentials

4. No Network Segmentation:
   ┌─────────────────────────────────────────────────┐
   │ All services on same network: milvus-net        │
   └─────────────────────────────────────────────────┘
   Risk: 🟡 MEDIUM - Lateral movement possible
```

### 7.2 Container Security Findings

| Finding | Severity | Status |
|---------|----------|--------|
| Running as non-root (API) | Info | ✅ Good |
| Default credentials (MinIO) | 🔴 Critical | ❌ Fix required |
| Default credentials (Grafana) | 🔴 Critical | ❌ Fix required |
| No resource limits | 🟡 Medium | ❌ Missing |
| No security contexts | 🟡 Medium | ❌ Missing |
| No read-only root FS | 🟢 Low | ❌ Missing |
| No network policies | 🟡 Medium | ❌ Missing |

---

## 8. Vulnerability Assessment

### 8.1 Critical Vulnerabilities

#### VULN-001: Hardcoded Credentials in Docker Compose
```
Severity: 🔴 CRITICAL
Location: infras/docker-compose.yml
Affected: MinIO, Grafana, PostgreSQL (Langfuse, Airflow)

Evidence:
- Line 32-33: MINIO_ACCESS_KEY/SECRET = minioadmin
- Line 119-120: GF_SECURITY_ADMIN_USER/PASSWORD = admin
- Line 139-141: POSTGRES credentials = langfuse
- Line 229-231: POSTGRES credentials = airflow

Impact: Unauthorized access to all data stores

Remediation:
1. Use Docker secrets or external secret manager
2. Generate unique passwords per environment
3. Rotate credentials regularly
```

#### VULN-002: No Encryption at Rest
```
Severity: 🔴 CRITICAL
Location: All data volumes

Affected:
- Milvus vectors (sensitive runbook content)
- Redis cache (user queries, responses)
- PostgreSQL (Langfuse traces, Airflow metadata)
- MinIO (Milvus objects)

Impact: Data breach if storage is compromised

Remediation:
1. Enable volume encryption (LUKS/dm-crypt)
2. Use cloud provider KMS
3. Implement application-level encryption
```

#### VULN-003: API Keys Never Expire
```
Severity: 🔴 CRITICAL
Location: src/sre_copilot/middleware/auth.py

Current implementation:
- API keys are static strings in environment
- No expiration mechanism
- No rotation support

Impact: Compromised keys remain valid indefinitely

Remediation:
1. Implement key expiration (30-90 days)
2. Add automatic rotation mechanism
3. Support key revocation API
4. Implement JWT tokens with refresh
```

### 8.2 High Severity Vulnerabilities

#### VULN-004: No Input Length Limit
```
Severity: 🟠 HIGH
Location: src/sre_copilot/schemas/retrieval.py

Current: query: str = Field(..., min_length=1)
Missing: max_length parameter

Impact: DoS via large payloads, excessive LLM costs

Remediation:
query: str = Field(..., min_length=1, max_length=4096)
```

#### VULN-005: No TLS for Internal Traffic
```
Severity: 🟠 HIGH
Location: All service-to-service communication

Affected paths:
- API → Milvus (port 19530)
- API → Redis (port 6379)
- API → OpenAI API
- Airflow → Data sources

Impact: Man-in-the-middle attacks

Remediation:
1. Deploy service mesh (Istio) with mTLS
2. Configure TLS for each service
```

---

## 9. Compliance Assessment

### 9.1 Compliance Gap Analysis

| Standard | Current Status | Key Gaps |
|----------|----------------|----------|
| **GDPR** | ⚠️ Partial | No data deletion API, retention policy |
| **SOC 2** | ❌ Not Ready | Missing encryption, access controls, audit |
| **ISO 27001** | ❌ Not Ready | No ISMS, risk assessment, policies |
| **HIPAA** | ❌ Not Ready | PHI not protected, no BAA support |
| **PCI-DSS** | ❌ Not Ready | Card data handling not compliant |
| **PDPA (Vietnam)** | ⚠️ Partial | PII handling exists, gaps in consent |

### 9.2 SOC 2 Readiness Checklist

| Control | Status | Gap |
|---------|--------|-----|
| Access Control | ⚠️ | No RBAC, weak auth |
| Encryption | ❌ | No encryption at rest |
| Monitoring | ✅ | Langfuse + OTEL |
| Change Management | ⚠️ | CI exists, no approval flow |
| Incident Response | ❌ | No IR plan |
| Vendor Management | ⚠️ | OpenAI dependency not documented |
| Risk Assessment | ❌ | No formal assessment |

---

## 10. Security Recommendations

### 10.1 Critical (Fix Immediately)

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 1 | Remove default credentials from docker-compose | S | Critical |
| 2 | Implement secrets management (Vault) | M | Critical |
| 3 | Add API key expiration and rotation | M | Critical |
| 4 | Enable encryption at rest for all volumes | M | Critical |

### 10.2 High Priority (Fix within 30 days)

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 5 | Add input length validation | S | High |
| 6 | Implement TLS for internal traffic | M | High |
| 7 | Add RBAC for multi-tenant access | L | High |
| 8 | Implement dependency vulnerability scanning | S | High |
| 9 | Add rate limiting per API key | S | High |

### 10.3 Medium Priority (Fix within 90 days)

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 10 | Add WAF (Web Application Firewall) | M | Medium |
| 11 | Implement network segmentation | M | Medium |
| 12 | Add output sanitization for LLM responses | S | Medium |
| 13 | Implement data retention policies | M | Medium |
| 14 | Add Vietnamese PII patterns | S | Medium |
| 15 | Create security incident response plan | M | Medium |

---

## 11. Ưu Điểm (ATTT View)

| # | Ưu điểm | Security Impact |
|---|---------|-----------------|
| 1 | **Guardrails framework** | Mitigates prompt injection |
| 2 | **PII detection** | Prevents data leakage |
| 3 | **Rate limiting** | Prevents DoS/abuse |
| 4 | **Audit logging** | Forensics capability |
| 5 | **Non-root container** | Limits container escape |
| 6 | **Input validation** | Reduces attack surface |
| 7 | **Fallback architecture** | Defense in depth |

---

## 12. Nhược Điểm (ATTT View)

| # | Nhược điểm | Risk Level | Priority |
|---|-----------|------------|----------|
| 1 | **No encryption at rest** | 🔴 Critical | P0 |
| 2 | **Default credentials** | 🔴 Critical | P0 |
| 3 | **No key expiration** | 🔴 Critical | P0 |
| 4 | **Weak authentication** | 🟠 High | P1 |
| 5 | **No RBAC** | 🟠 High | P1 |
| 6 | **No TLS internal** | 🟠 High | P1 |
| 7 | **No WAF** | 🟡 Medium | P2 |
| 8 | **No vuln scanning** | 🟡 Medium | P2 |
| 9 | **No IR plan** | 🟡 Medium | P2 |

---

## 13. Security Maturity Assessment

```
Security Maturity Model (1-5 scale):

┌─────────────────────────────────────────────────────────────────────┐
│ Domain                    │ Current │ Target │ Gap                 │
├───────────────────────────┼─────────┼────────┼─────────────────────┤
│ Identity & Access         │   2     │   4    │ RBAC, SSO needed    │
│ Data Protection           │   1     │   4    │ Encryption needed   │
│ Infrastructure Security   │   2     │   4    │ Network seg needed  │
│ Application Security      │   3     │   4    │ Output sanit needed │
│ Security Operations       │   2     │   4    │ IR plan needed      │
│ Compliance                │   1     │   4    │ Major gaps          │
├───────────────────────────┼─────────┼────────┼─────────────────────┤
│ OVERALL                   │  1.8    │   4    │ Significant work    │
└─────────────────────────────────────────────────────────────────────┘

Legend: 1=Initial, 2=Developing, 3=Defined, 4=Managed, 5=Optimized
```

---

## 14. Kết Luận

Dự án AI SRE Copilot có một số security controls tốt ở application layer (guardrails, PII detection) nhưng có **3 lỗ hổng Critical** cần fix ngay:

1. **Default credentials** trong docker-compose
2. **Không có encryption at rest**
3. **API keys không có expiration**

**Recommendation:** Không deploy production cho đến khi fix các lỗ hổng Critical và High.

---

*Phân tích bởi: AI Security Expert*
*Phiên bản: 1.0*
*Classification: Internal Use Only*
