# src/mcp/tools/analytics.py
import logging
import os
import json
from datetime import datetime
from typing import List
from configs.settings import settings
import prometheus_client
from mcp.schemas.schemas import AnalyticsLogRequest, AnalyticsLogResponse, AnalyticsMetricsRequest, AnalyticsMetricsResponse, MetricItem

# Подключение redis.asyncio:
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# (Опционально) Prometheus-метрики для аналитики. Проверьте, чтобы не дублировались имена с другими частями.
EVENT_LOG_COUNTER = prometheus_client.Counter(
    'analytics_events_logged_total', 'Number of analytics events logged', ['event']
)
# Можно добавить гистограмму задержек, если требуется:
# EVENT_LOG_LATENCY = prometheus_client.Histogram(...)


class AnalyticsTool:
    """
    Инструмент для логирования событий: Redis + (Prometheus для метрик HTTP/other).
    """
    def __init__(self):
        try:
            REDIS_URL = settings.REDIS_URL  # из Pydantic settings
            self.redis = aioredis.from_url(str(REDIS_URL))
            self.event_key_prefix = 'analytics:event:'
            logger.info(f"AnalyticsTool: connected to Redis at {REDIS_URL}")
        except Exception as e:
            logger.warning(f"AnalyticsTool: cannot connect to Redis: {e}", exc_info=True)
            self.redis = None

    async def log_event(self, req: AnalyticsLogRequest) -> AnalyticsLogResponse:
        ts = datetime.utcnow().isoformat() + 'Z'
        record = {
            'timestamp': ts,
            'event': req.event,
            'user_id': req.user_id,
            'session_id': req.session_id,
            'metadata': req.metadata
        }
        if not self.redis:
            logger.warning("AnalyticsTool.log_event: Redis unavailable")
            return AnalyticsLogResponse(success=False, timestamp=ts)
        key = f"{self.event_key_prefix}{req.event}"
        try:
            # Храним JSON-строку
            await self.redis.rpush(key, json.dumps(record))
            # Опционально: задать TTL, например:
            # await self.redis.expire(key, settings.ANALYTICS_TTL_SECONDS)
            logger.debug(f"AnalyticsTool.log_event: recorded event {req.event} at {ts}")
            # Prometheus-счетчик
            try:
                EVENT_LOG_COUNTER.labels(event=req.event).inc()
            except Exception:
                pass
            return AnalyticsLogResponse(success=True, timestamp=ts)
        except Exception as e:
            logger.error(f"AnalyticsTool.log_event: error writing to Redis: {e}", exc_info=True)
            return AnalyticsLogResponse(success=False, timestamp=ts)

    async def get_metrics(self, req: AnalyticsMetricsRequest) -> AnalyticsMetricsResponse:
        results: List[MetricItem] = []
        now_ts = datetime.utcnow().isoformat() + 'Z'
        if not self.redis:
            logger.warning("AnalyticsTool.get_metrics: Redis unavailable")
            return AnalyticsMetricsResponse(metrics=[], calculated_at=now_ts)
        for name in req.metric_names or []:
            key = f"{self.event_key_prefix}{name}"
            try:
                records = await self.redis.lrange(key, 0, -1)
            except Exception as e:
                logger.error(f"AnalyticsTool.get_metrics: cannot lrange {key}: {e}", exc_info=True)
                continue
            count = 0
            for rec_bytes in records:
                try:
                    rec = json.loads(rec_bytes)
                    ts = datetime.fromisoformat(rec['timestamp'].replace('Z', '+00:00'))
                    if req.start_time:
                        start = datetime.fromisoformat(req.start_time.replace('Z', '+00:00'))
                        if ts < start:
                            continue
                    if req.end_time:
                        end = datetime.fromisoformat(req.end_time.replace('Z', '+00:00'))
                        if ts > end:
                            continue
                    count += 1
                except Exception:
                    continue
            results.append(MetricItem(name=name, value=count))
        calc_ts = datetime.utcnow().isoformat() + 'Z'
        logger.debug(f"AnalyticsTool.get_metrics at {calc_ts}, metrics returned: {results}")
        return AnalyticsMetricsResponse(metrics=results, calculated_at=calc_ts)
