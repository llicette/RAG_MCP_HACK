import logging
from typing import List, Dict, Any
from datetime import datetime
from src.mcp.schemas.schemas import AnalyticsLogRequest, AnalyticsLogResponse, AnalyticsMetricsRequest, AnalyticsMetricsResponse, MetricItem
import os
import aioredis
import json
import prometheus_client

logger = logging.getLogger(__name__)

# Prometheus метрики
REQUEST_COUNT = prometheus_client.Counter('mcp_request_count', 'Number of MCP requests', ['endpoint'])
REQUEST_LATENCY = prometheus_client.Histogram('mcp_request_latency_seconds', 'Latency MCP endpoints', ['endpoint'])

class AnalyticsTool:
    """
    Инструмент для логирования событий: Redis + Prometheus
    """
    def __init__(self):
        # Redis клиент
        REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379/0')
        self.redis = aioredis.from_url(REDIS_URL)
        # Namespace для событий
        self.event_key_prefix = 'analytics:event:'

    async def log_event(self, req: AnalyticsLogRequest) -> AnalyticsLogResponse:
        ts = datetime.utcnow().isoformat() + 'Z'
        record = {
            'timestamp': ts,
            'event': req.event,
            'user_id': req.user_id,
            'session_id': req.session_id,
            'metadata': req.metadata
        }
        # Храним в Redis список
        key = f"{self.event_key_prefix}{req.event}"
        await self.redis.rpush(key, json.dumps(record))
        # Устанавливаем TTL для старых записей, если нужно
        logger.debug(f"AnalyticsTool.log_event: recorded event {req.event} at {ts}")
        return AnalyticsLogResponse(success=True, timestamp=ts)

    async def get_metrics(self, req: AnalyticsMetricsRequest) -> AnalyticsMetricsResponse:
        results: List[MetricItem] = []
        # Для каждого имени метрики считываем Redis
        for name in req.metric_names or []:
            key = f"{self.event_key_prefix}{name}"
            records = await self.redis.lrange(key, 0, -1)
            # Фильтрация по времени, если задана
            count = 0
            for rec_bytes in records:
                try:
                    rec = json.loads(rec_bytes)
                    ts = datetime.fromisoformat(rec['timestamp'].replace('Z', '+00:00'))
                    if req.start_time:
                        start = datetime.fromisoformat(req.start_time.replace('Z', '+00:00'))
                        if ts < start: continue
                    if req.end_time:
                        end = datetime.fromisoformat(req.end_time.replace('Z', '+00:00'))
                        if ts > end: continue
                    count += 1
                except Exception:
                    continue
            results.append(MetricItem(name=name, value=count))
        calc_ts = datetime.utcnow().isoformat() + 'Z'
        logger.debug(f"AnalyticsTool.get_metrics at {calc_ts}, metrics returned: {results}")
        return AnalyticsMetricsResponse(metrics=results, calculated_at=calc_ts)