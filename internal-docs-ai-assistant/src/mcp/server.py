import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from mcp.schemas.schemas import (
    DocumentSearchRequest, DocumentSearchResponse,
    KnowledgeQueryRequest, KnowledgeQueryResponse,
    AnalyticsLogRequest, AnalyticsLogResponse,
    AnalyticsMetricsRequest, AnalyticsMetricsResponse
)
from mcp.tools.document_search import DocumentSearchTool
from mcp.tools.knowledge_base import KnowledgeBaseTool
from mcp.tools.analytics import AnalyticsTool
from mcp.db import init_db
import prometheus_client
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import time
from configs.settings import settings


app = FastAPI(title="MCP Server for AI Documentation Assistant")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация инструментов
@app.on_event("startup")
async def startup_event():
    configure_logging()
    # Инициализация DB
    await init_db()
    # Инициализация инструментов и сохранение в state
    app.state.document_search_tool = DocumentSearchTool()
    app.state.knowledge_base_tool = KnowledgeBaseTool()
    app.state.analytics_tool = AnalyticsTool()
    logger = logging.getLogger(__name__)
    logger.info("MCP Server startup complete")

# Dependency: получить user_id/session_id
async def get_context(user_id: str = Header(None), session_id: str = Header(None)):
    return {"user_id": user_id, "session_id": session_id}

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    endpoint = request.url.path.replace('/', '_').strip('_') or 'root'
    prometheus_client.Counter('mcp_requests_total', 'Count of requests', ['endpoint']).labels(endpoint=endpoint).inc()
    prometheus_client.Histogram('mcp_request_latency_seconds', 'Request latency', ['endpoint']).labels(endpoint=endpoint).observe(latency)
    return response

@app.post("/document_search", response_model=DocumentSearchResponse)
async def document_search_endpoint(
    req: DocumentSearchRequest,
    ctx: dict = Depends(get_context)
):
    tool: DocumentSearchTool = app.state.document_search_tool
    # Логируем событие
    await app.state.analytics_tool.log_event(AnalyticsLogRequest(
        event="document_search_called",
        user_id=ctx.get("user_id"),
        session_id=ctx.get("session_id"),
        metadata={"query": req.query}
    ))
    try:
        resp = await tool.search(req)
        return resp
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка document_search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/document_index")
async def document_index_endpoint(
    doc_id: str, title: str, content: str, metadata: dict,
    ctx: dict = Depends(get_context)
):
    tool: DocumentSearchTool = app.state.document_search_tool
    try:
        await tool.index_document(doc_id, title, content, metadata)
        return {"success": True}
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка document_index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/document_delete")
async def document_delete_endpoint(
    doc_id: str,
    ctx: dict = Depends(get_context)
):
    tool: DocumentSearchTool = app.state.document_search_tool
    try:
        await tool.delete_document(doc_id)
        return {"success": True}
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка document_delete: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge_query", response_model=KnowledgeQueryResponse)
async def knowledge_query_endpoint(
    req: KnowledgeQueryRequest,
    ctx: dict = Depends(get_context)
):
    tool: KnowledgeBaseTool = app.state.knowledge_base_tool
    await app.state.analytics_tool.log_event(AnalyticsLogRequest(
        event="knowledge_query_called",
        user_id=ctx.get("user_id"),
        session_id=ctx.get("session_id"),
        metadata={"topic": req.topic}
    ))
    try:
        resp = await tool.query(req)
        return resp
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка knowledge_query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge_add")
async def knowledge_add_endpoint(entry: KnowledgeEntry, ctx: dict = Depends(get_context)):
    tool: KnowledgeBaseTool = app.state.knowledge_base_tool
    try:
        success = await tool.add_entry(entry)
        return {"success": success}
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка knowledge_add: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge_remove")
async def knowledge_remove_endpoint(entry_id: str, ctx: dict = Depends(get_context)):
    tool: KnowledgeBaseTool = app.state.knowledge_base_tool
    try:
        success = await tool.remove_entry(entry_id)
        return {"success": success}
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка knowledge_remove: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/log", response_model=AnalyticsLogResponse)
async def analytics_log_endpoint(
    req: AnalyticsLogRequest,
    ctx: dict = Depends(get_context)
):
    try:
        resp = await app.state.analytics_tool.log_event(req)
        return resp
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка analytics_log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/metrics", response_model=AnalyticsMetricsResponse)
async def analytics_metrics_endpoint(
    req: AnalyticsMetricsRequest,
    ctx: dict = Depends(get_context)
):
    try:
        resp = await app.state.analytics_tool.get_metrics(req)
        return resp
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка analytics_metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.get("/healthz")
async def health_check():
    results = {}
    # Postgres
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        results['postgres'] = True
    except Exception:
        results['postgres'] = False
    # Redis
    try:
        pong = await app.state.analytics_tool.redis.ping()
        results['redis'] = True
    except Exception:
        results['redis'] = False
    # Qdrant
    try:
        _ = app.state.document_search_tool.qdrant.get_collections()
        results['qdrant'] = True
    except Exception:
        results['qdrant'] = False
    status = all(results.values())
    return {"status": "ok" if status else "degraded", **results}


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
    )

if __name__ == "__main__":
    configure_logging()
    uvicorn.run("src.mcp.server:app", host="0.0.0.0", port=8000, log_level="info")