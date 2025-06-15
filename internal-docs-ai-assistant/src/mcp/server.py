import logging
import uvicorn
import time
import prometheus_client
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from fastapi import FastAPI, HTTPException, Depends, Request, Header, Response, Body
from fastapi.middleware.cors import CORSMiddleware

from configs.settings import settings

from mcp.schemas.schemas import (
    DocumentSearchRequest, DocumentSearchResponse,
    DocumentIndexRequest, DocumentDeleteRequest,
    KnowledgeQueryRequest, KnowledgeQueryResponse,
    AnalyticsLogRequest, AnalyticsLogResponse,
    AnalyticsMetricsRequest, AnalyticsMetricsResponse,
    HealthResponse,
    KnowledgeEntry  # импорт модели
)
from mcp.tools.document_search import DocumentSearchTool
from mcp.tools.knowledge_base import KnowledgeBaseTool
from mcp.tools.analytics import AnalyticsTool
from mcp.db import init_db, AsyncSessionLocal  # импорт AsyncSessionLocal
from sqlalchemy import text

app = FastAPI(title="MCP Server for AI Documentation Assistant")

# Конфигурируем логирование
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
    )

# Prometheus метрики: вынесем метрики глобально, чтобы не пересоздавать каждый запрос
REQUEST_COUNTER = prometheus_client.Counter(
    'mcp_requests_total', 'Count of requests', ['endpoint']
)
REQUEST_LATENCY = prometheus_client.Histogram(
    'mcp_request_latency_seconds', 'Request latency', ['endpoint']
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # при необходимости ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    configure_logging()
    logger = logging.getLogger(__name__)
    # Инициализация БД
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"init_db failed: {e}", exc_info=True)
        # Решите: если БД критична, можно завершить запуск, либо продолжать с degraded.
    # Инициализация инструментов
    # DocumentSearchTool
    try:
        app.state.document_search_tool = DocumentSearchTool()
        logger.info("DocumentSearchTool initialized")
    except Exception as e:
        logger.warning(f"Cannot init DocumentSearchTool: {e}", exc_info=True)
        app.state.document_search_tool = None
    # KnowledgeBaseTool
    try:
        app.state.knowledge_base_tool = KnowledgeBaseTool()
        logger.info("KnowledgeBaseTool initialized")
    except Exception as e:
        logger.warning(f"Cannot init KnowledgeBaseTool: {e}", exc_info=True)
        app.state.knowledge_base_tool = None
    # AnalyticsTool
    try:
        app.state.analytics_tool = AnalyticsTool()
        logger.info("AnalyticsTool initialized")
    except Exception as e:
        logger.warning(f"Cannot init AnalyticsTool: {e}", exc_info=True)
        app.state.analytics_tool = None

    logger.info("MCP Server startup complete")

# Dependency: получить user_id/session_id из заголовков
async def get_context(user_id: str = Header(None), session_id: str = Header(None)):
    return {"user_id": user_id, "session_id": session_id}

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    endpoint = request.url.path.replace('/', '_').strip('_') or 'root'
    REQUEST_COUNTER.labels(endpoint=endpoint).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
    return response

@app.post("/document_search", response_model=DocumentSearchResponse)
async def document_search_endpoint(
    req: DocumentSearchRequest,
    ctx: dict = Depends(get_context)
):
    tool = app.state.document_search_tool
    if tool is None:
        raise HTTPException(status_code=503, detail="DocumentSearchTool not available")
    # Лог события аналитики
    if app.state.analytics_tool:
        try:
            await app.state.analytics_tool.log_event(AnalyticsLogRequest(
                event="document_search_called",
                user_id=ctx.get("user_id"),
                session_id=ctx.get("session_id"),
                metadata={"query": req.query}
            ))
        except Exception:
            logging.getLogger(__name__).warning("analytics log_event failed", exc_info=True)
    try:
        resp = await tool.search(req)
        return resp
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка document_search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error in document_search")

@app.post("/document_index")
async def document_index_endpoint(
    req: DocumentIndexRequest,
    ctx: dict = Depends(get_context)
):
    tool = app.state.document_search_tool
    if tool is None:
        raise HTTPException(status_code=503, detail="DocumentSearchTool not available")
    try:
        await tool.index_document(req.doc_id, req.title, req.content, req.metadata or {})
        # Лог события индексирования?
        if app.state.analytics_tool:
            try:
                await app.state.analytics_tool.log_event(AnalyticsLogRequest(
                    event="document_index_called",
                    user_id=ctx.get("user_id"),
                    session_id=ctx.get("session_id"),
                    metadata={"doc_id": req.doc_id}
                ))
            except:
                pass
        return {"success": True}
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка document_index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error in document_index")

@app.post("/document_delete")
async def document_delete_endpoint(
    req: DocumentDeleteRequest,
    ctx: dict = Depends(get_context)
):
    tool = app.state.document_search_tool
    if tool is None:
        raise HTTPException(status_code=503, detail="DocumentSearchTool not available")
    try:
        await tool.delete_document(req.doc_id)
        if app.state.analytics_tool:
            try:
                await app.state.analytics_tool.log_event(AnalyticsLogRequest(
                    event="document_delete_called",
                    user_id=ctx.get("user_id"),
                    session_id=ctx.get("session_id"),
                    metadata={"doc_id": req.doc_id}
                ))
            except:
                pass
        return {"success": True}
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка document_delete: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error in document_delete")

@app.post("/knowledge_query", response_model=KnowledgeQueryResponse)
async def knowledge_query_endpoint(
    req: KnowledgeQueryRequest,
    ctx: dict = Depends(get_context)
):
    tool = app.state.knowledge_base_tool
    if tool is None:
        raise HTTPException(status_code=503, detail="KnowledgeBaseTool not available")
    if app.state.analytics_tool:
        try:
            await app.state.analytics_tool.log_event(AnalyticsLogRequest(
                event="knowledge_query_called",
                user_id=ctx.get("user_id"),
                session_id=ctx.get("session_id"),
                metadata={"topic": req.topic}
            ))
        except Exception:
            logging.getLogger(__name__).warning("analytics log_event failed", exc_info=True)
    try:
        resp = await tool.query(req)
        return resp
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка knowledge_query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error in knowledge_query")

@app.post("/knowledge_add")
async def knowledge_add_endpoint(
    entry: KnowledgeEntry,
    ctx: dict = Depends(get_context)
):
    tool = app.state.knowledge_base_tool
    if tool is None:
        raise HTTPException(status_code=503, detail="KnowledgeBaseTool not available")
    try:
        success = await tool.add_entry(entry)
        if app.state.analytics_tool:
            try:
                await app.state.analytics_tool.log_event(AnalyticsLogRequest(
                    event="knowledge_add_called",
                    user_id=ctx.get("user_id"),
                    session_id=ctx.get("session_id"),
                    metadata={"entry_id": entry.id if entry.id else None}
                ))
            except:
                pass
        return {"success": success}
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка knowledge_add: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error in knowledge_add")

# Для удаления: можно создать схему KnowledgeDeleteRequest или Body параметр
@app.post("/knowledge_remove")
async def knowledge_remove_endpoint(
    entry_id: str = Body(..., embed=True),  # ожидаем JSON {"entry_id": "..."}
    ctx: dict = Depends(get_context)
):
    tool = app.state.knowledge_base_tool
    if tool is None:
        raise HTTPException(status_code=503, detail="KnowledgeBaseTool not available")
    try:
        success = await tool.remove_entry(entry_id)
        if app.state.analytics_tool:
            try:
                await app.state.analytics_tool.log_event(AnalyticsLogRequest(
                    event="knowledge_remove_called",
                    user_id=ctx.get("user_id"),
                    session_id=ctx.get("session_id"),
                    metadata={"entry_id": entry_id}
                ))
            except:
                pass
        return {"success": success}
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка knowledge_remove: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error in knowledge_remove")

@app.post("/analytics/log", response_model=AnalyticsLogResponse)
async def analytics_log_endpoint(
    req: AnalyticsLogRequest,
    ctx: dict = Depends(get_context)
):
    tool = app.state.analytics_tool
    if tool is None:
        raise HTTPException(status_code=503, detail="AnalyticsTool not available")
    try:
        resp = await tool.log_event(req)
        return resp
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка analytics_log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error in analytics_log")

@app.post("/analytics/metrics", response_model=AnalyticsMetricsResponse)
async def analytics_metrics_endpoint(
    req: AnalyticsMetricsRequest,
    ctx: dict = Depends(get_context)
):
    tool = app.state.analytics_tool
    if tool is None:
        raise HTTPException(status_code=503, detail="AnalyticsTool not available")
    try:
        resp = await tool.get_metrics(req)
        return resp
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка analytics_metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error in analytics_metrics")

@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    results = {}
    # Postgres
    ok_pg = False
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        ok_pg = True
    except Exception:
        ok_pg = False
    results['postgres'] = ok_pg
    # Redis
    ok_redis = False
    if app.state.analytics_tool and getattr(app.state.analytics_tool, "redis", None):
        try:
            pong = await app.state.analytics_tool.redis.ping()
            ok_redis = True
        except Exception:
            ok_redis = False
    results['redis'] = ok_redis
    # Qdrant
    ok_qdrant = False
    if app.state.document_search_tool and getattr(app.state.document_search_tool, "qdrant", None):
        try:
            _ = app.state.document_search_tool.qdrant.get_collections()
            ok_qdrant = True
        except Exception:
            ok_qdrant = False
    results['qdrant'] = ok_qdrant

    overall = all(results.values())
    status = "ok" if overall else "degraded"
    return HealthResponse(status=status)  # либо можно вернуть details: {**results}

if __name__ == "__main__":
    configure_logging()
    # Запуск: если файл находится в папке src/mcp/server.py, то:
    # uvicorn.run("mcp.server:app", host="0.0.0.0", port=8000, log_level="info")
    # или, в зависимости от точки старта, "src.mcp.server:app"
    uvicorn.run("mcp.server:app", host="0.0.0.0", port=8000, log_level="info")
