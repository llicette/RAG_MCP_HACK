import logging
from typing import List, Dict, Any
from src.mcp.schemas.schemas import DocumentSearchRequest, DocumentSearchResponse, DocumentItem
from src.mcp.db import AsyncSessionLocal, DocumentRecord
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import os
import asyncio
import json

logger = logging.getLogger(__name__)

class DocumentSearchTool:
    """
    Инструмент для поиска документов.
    Семантический поиск: Qdrant + SentenceTransformer.
    Полнотекстовый поиск: Postgres full-text search.
    """
    def __init__(self):
        # Инициализация клиентa Qdrant
        QDRANT_URL = os.getenv('QDRANT_URL', 'http://qdrant:6333')
        QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', None)
        self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # Название коллекции
        self.collection_name = os.getenv('QDRANT_COLLECTION', 'documents')
        # Инициализация SentenceTransformer
        model_name = os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
        self.embedder = SentenceTransformer(model_name)

        # Создаем коллекцию, если не существует
        try:
            self.qdrant.get_collection(collection_name=self.collection_name)
        except Exception:
            # Предполагаем размер embedding 384
            self.qdrant.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(size=self.embedder.get_sentence_embedding_dimension(), distance=rest.Distance.COSINE),
            )
        logger.info("DocumentSearchTool initialized: Qdrant collection '%s'", self.collection_name)

    async def search(self, req: DocumentSearchRequest) -> DocumentSearchResponse:
        logger.debug(f"DocumentSearchTool.search called with query={req.query}, filters={req.filters}, top_k={req.top_k}")
        # Семантический поиск
        query_embedding = await asyncio.to_thread(self.embedder.encode, req.query)
        # Поиск в Qdrant
        search_result = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=req.top_k,
            with_payload=True
        )
        documents: List[DocumentItem] = []
        ids = [hit.id for hit in search_result]
        # Получаем полные данные из Postgres
        async with AsyncSessionLocal() as session:
            for hit in search_result:
                doc_id = str(hit.id)
                # Извлечение метаданных из payload
                payload = hit.payload or {}
                # Получаем title и snippet или full content
                record = await session.get(DocumentRecord, doc_id)
                if record:
                    snippet = None
                    # Можно сформировать snippet: первые 200 символов
                    snippet = record.content[:200] + '...'
                    documents.append(DocumentItem(
                        id=doc_id,
                        title=record.title,
                        content=snippet,
                        metadata={**payload, **record.metadata}
                    ))
        total = len(documents)
        logger.debug(f"DocumentSearchTool.search returning {total} documents")
        return DocumentSearchResponse(documents=documents, total=total)

    async def index_document(self, doc_id: str, title: str, content: str, metadata: Dict[str, Any]):
        """Индексируем документ: сохраняем в Postgres и Qdrant"""
        # Сохраняем запись в Postgres
        async with AsyncSessionLocal() as session:
            record = await session.get(DocumentRecord, doc_id)
            if not record:
                record = DocumentRecord(id=doc_id, title=title, content=content, metadata=metadata)
                session.add(record)
            else:
                record.title = title
                record.content = content
                record.metadata = metadata
            await session.commit()
        # Индексируем в Qdrant
        # Генерируем embedding
        embedding = await asyncio.to_thread(self.embedder.encode, content)
        # upsert
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[rest.PointStruct(id=doc_id, vector=embedding.tolist(), payload=metadata)]
        )
        logger.info(f"Document indexed: {doc_id}")

    async def delete_document(self, doc_id: str):
        # Удаляем из Postgres
        async with AsyncSessionLocal() as session:
            record = await session.get(DocumentRecord, doc_id)
            if record:
                await session.delete(record)
                await session.commit()
        # Удаляем из Qdrant
        self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=rest.PointIdsList([doc_id])
        )
        logger.info(f"Document deleted: {doc_id}")