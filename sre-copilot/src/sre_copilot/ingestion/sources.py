from __future__ import annotations

import datetime as dt
from typing import List

import httpx

from sre_copilot.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_window(settings, now: dt.datetime | None = None) -> tuple[str, str]:
    now = now or dt.datetime.utcnow()
    end = now
    start = end - dt.timedelta(minutes=settings.ingest_lookback_minutes)
    return start.isoformat() + "Z", end.isoformat() + "Z"


async def fetch_loki_logs(settings) -> List[str]:
    if not settings.loki_url:
        return []
    start, end = calculate_window(settings)
    query = settings.loki_query or '{job!=""}'
    params = {"query": query, "start": start, "end": end}
    headers = {}
    if getattr(settings, "loki_tenant", None):
        headers["X-Scope-OrgID"] = settings.loki_tenant
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            logger.info(
                "Loki request %s/loki/api/v1/query_range params=%s headers=%s",
                settings.loki_url,
                params,
                {"X-Scope-OrgID": headers.get("X-Scope-OrgID")} if headers else {},
            )
            resp = await client.get(f"{settings.loki_url}/loki/api/v1/query_range", params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("data", {}).get("result", [])
            logger.info("Loki returned %s result streams", len(result))
            logs = []
            for stream in result:
                labels = stream.get("stream", {}) or {}
                label_str = ", ".join(f"{k}={v}" for k, v in labels.items())
                for value in stream.get("values", []):
                    if len(value) > 1:
                        ts = value[0]
                        line = value[1]
                        if label_str:
                            logs.append(f"[{ts}] {label_str} {line}")
                        else:
                            logs.append(f"[{ts}] {line}")
            return logs
    except Exception as exc:  # pragma: no cover
        logger.warning("Loki fetch failed: %s", exc)
        return []


async def fetch_prometheus_metrics(settings) -> List[str]:
    if not settings.prom_url:
        return []
    start, end = calculate_window(settings)
    params = {
        "query": settings.prom_query,
        "start": start,
        "end": end,
        "step": "60",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{settings.prom_url}/api/v1/query_range", params=params)
            resp.raise_for_status()
            results = resp.json().get("data", {}).get("result", [])
            snippets = []
            for series in results:
                metric = series.get("metric", {})
                values = series.get("values", [])
                labels = ", ".join(f"{k}={v}" for k, v in metric.items())
                for ts, val in values:
                    snippets.append(f"{ts} {labels} value={val}")
            return snippets
    except Exception as exc:  # pragma: no cover
        logger.warning("Prometheus fetch failed: %s", exc)
        return []


async def fetch_tempo_traces(settings) -> List[str]:
    if not settings.tempo_url:
        logger.info("Tempo URL missing; skipping tempo fetch")
        return []
    params = {"limit": 20}
    if settings.tempo_service:
        params["service"] = settings.tempo_service
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            logger.info("Tempo request %s/api/search params=%s", settings.tempo_url, params)
            resp = await client.get(f"{settings.tempo_url}/api/search", params=params)
            resp.raise_for_status()
            data = resp.json()
            logger.info("Tempo response: %s", data)
            traces = data.get("traces") or data.get("traces_search", [])
            snippets = []
            for trace in traces:
                trace_id = trace.get("traceID") or trace.get("traceId")
                duration = trace.get("durationMs") or trace.get("duration")
                svc = trace.get("serviceName") or settings.tempo_service
                snippets.append(f"trace {trace_id} service={svc} duration={duration}")
            return snippets
    except Exception as exc:  # pragma: no cover
        logger.warning("Tempo fetch failed: %s", exc)
        return []
