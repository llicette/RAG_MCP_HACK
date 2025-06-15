# src/mcp/schemas/schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Документы ---
class DocumentItem(BaseModel):
    id: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = {}

class DocumentSearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1)
    filters: Optional[Dict[str, Any]] = None  # на будущее

class DocumentSearchResponse(BaseModel):
    documents: List[DocumentItem]
    total: int

# --- База знаний ---
class KnowledgeEntry(BaseModel):
    id: Optional[str] = None
    title: str
    summary: str
    content: str
    tags: Optional[List[str]] = []

class KnowledgeQueryRequest(BaseModel):
    topic: str
    context: Optional[str] = None

class KnowledgeQueryResponse(BaseModel):
    entries: List[KnowledgeEntry]
    total: int

# --- Аналитика ---
class AnalyticsLogRequest(BaseModel):
    event: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class AnalyticsLogResponse(BaseModel):
    success: bool
    timestamp: str

class MetricItem(BaseModel):
    name: str
    value: int

class AnalyticsMetricsRequest(BaseModel):
    metric_names: Optional[List[str]] = None
    start_time: Optional[str] = None  # ISO format
    end_time: Optional[str] = None

class AnalyticsMetricsResponse(BaseModel):
    metrics: List[MetricItem]
    calculated_at: str

# --- для DocumentIndex и DocumentDelete ---
class DocumentIndexRequest(BaseModel):
    doc_id: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = {}

class DocumentDeleteRequest(BaseModel):
    doc_id: str

# --- Можно добавить ответ для health check ---
class HealthResponse(BaseModel):
    status: str
