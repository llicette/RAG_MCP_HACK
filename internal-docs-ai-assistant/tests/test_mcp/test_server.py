# tests/test_mcp/test_server.py

import pytest
from fastapi.testclient import TestClient
from mcp.server import app

class DummyDocTool:
    async def search(self, req):
        from src.mcp.schemas.schemas import DocumentSearchResponse, DocumentItem
        items = [DocumentItem(id="1", title="T1", content="C1", metadata={})]
        return DocumentSearchResponse(documents=items, total=1)
    async def index_document(self, doc_id, title, content, metadata):
        pass
    async def delete_document(self, doc_id):
        pass

class DummyKB:
    async def query(self, req):
        from src.mcp.schemas.schemas import KnowledgeQueryResponse, KnowledgeEntry
        entries = [KnowledgeEntry(id="k1", title="Title", summary="Sum", content="Cont", tags=[])]
        return KnowledgeQueryResponse(entries=entries, total=1)
    async def add_entry(self, entry):
        return True
    async def remove_entry(self, entry_id):
        return True

class DummyAnalytics:
    async def log_event(self, req):
        from src.mcp.schemas.schemas import AnalyticsLogResponse
        return AnalyticsLogResponse(success=True, timestamp="2025-01-01T00:00:00Z")
    async def get_metrics(self, req):
        from src.mcp.schemas.schemas import AnalyticsMetricsResponse, MetricItem
        return AnalyticsMetricsResponse(metrics=[MetricItem(name="e", value=1)], calculated_at="2025-01-01T00:00:00Z")

@pytest.fixture(autouse=True)
def override_tools(monkeypatch):
    # Перед тестами заменяем app.state инструменты
    from src.mcp.server import app
    monkeypatch.setattr(app.state, "document_search_tool", DummyDocTool())
    monkeypatch.setattr(app.state, "knowledge_base_tool", DummyKB())
    monkeypatch.setattr(app.state, "analytics_tool", DummyAnalytics())
    yield

@pytest.fixture
def client():
    return TestClient(app)

def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_document_search(client):
    resp = client.post("/document_search", json={"query": "q", "top_k": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert isinstance(data["documents"], list)
