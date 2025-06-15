# src/mcp/tools/document_search.py
import logging
import os
import asyncio
from typing import List, Dict, Any

from mcp.schemas.schemas import DocumentSearchRequest, DocumentSearchResponse, DocumentItem
from mcp.db import AsyncSessionLocal, DocumentRecord
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from configs.settings import settings

logger = logging.getLogger(__name__)

class DocumentSearchTool:
    """
    Инструмент для поиска документов.
    Семантический поиск: Qdrant + SentenceTransformer.
    Полнотекстовый поиск: Postgres full-text search (можно добавить).
    """
    def __init__(self):
        # Qdrant и SentenceTransformer
        try:
            QDRANT_URL = os.getenv('QDRANT_URL')
            QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', None)
            self.qdrant = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    prefer_grpc=False
                )
            self.collection_name = os.getenv('QDRANT_COLLECTION', 'documents')
            model_name = settings.EMBEDDING_MODEL_NAME  # "sentence-transformers/all-MiniLM-L6-v2"
            # Загружаем модель (может занять время)
            self.embedder = SentenceTransformer(model_name)
            # Проверяем или создаём коллекцию
            try:
                self.qdrant.get_collection(collection_name=self.collection_name)
            except Exception:
                dim = self.embedder.get_sentence_embedding_dimension()
                self.qdrant.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
                )
            logger.info("DocumentSearchTool: Qdrant initialized")
        except Exception as e:
            logger.warning(f"DocumentSearchTool: не удалось инициализировать Qdrant: {e}", exc_info=True)
            self.qdrant = None
            self.embedder = None
            self.collection_name = None

    async def search(self, req: DocumentSearchRequest) -> DocumentSearchResponse:
        """
        Выполняет поиск: сначала семантический через Qdrant, если есть; 
        затем (опционально) полнотекстовый через Postgres.
        """
        logger.debug(f"DocumentSearchTool.search query={req.query!r}, top_k={req.top_k}")
        documents: List[DocumentItem] = []

        query_text = (req.query or "").strip()
        if not query_text:
            logger.info("DocumentSearchTool.search: пустой запрос, возвращаем пустой список")
            return DocumentSearchResponse(documents=[], total=0)

        # Семантический поиск через Qdrant
        if self.qdrant and self.embedder:
            try:
                # Получаем embedding в фоне
                query_embedding = await asyncio.to_thread(self.embedder.encode, query_text)
                # Выполняем поиск в фоне
                search_result = await asyncio.to_thread(
                    self.qdrant.search,
                    collection_name=self.collection_name,
                    query_vector=query_embedding.tolist(),
                    limit=req.top_k,
                    with_payload=True
                )
                # Если коллекция есть, search_result — список hits
                # Обрабатываем каждый hit
                async with AsyncSessionLocal() as session:
                    for hit in search_result:
                        doc_id = str(hit.id)
                        payload = hit.payload or {}
                        try:
                            record = await session.get(DocumentRecord, doc_id)
                        except Exception as e:
                            logger.error(f"Error fetching DocumentRecord {doc_id}: {e}", exc_info=True)
                            record = None
                        if record:
                            snippet = (record.content[:200] + '...') if record.content else ''
                            documents.append(DocumentItem(
                                id=doc_id,
                                title=record.title,
                                content=snippet,
                                metadata={**payload, **(record.meta or {})}
                            ))
                        else:
                            # Если есть точка в Qdrant, но нет в Postgres, можно удалить её из Qdrant:
                            logger.warning(f"DocumentSearchTool: Qdrant point {doc_id} отсутствует в Postgres, удаляем из Qdrant")
                            # Удаляем в фоне:
                            await asyncio.to_thread(
                                self.qdrant.delete,
                                collection_name=self.collection_name,
                                points_selector=rest.PointIdsList([doc_id])
                            )
                logger.debug(f"DocumentSearchTool.search: found {len(documents)} docs via Qdrant")
            except Exception as e:
                logger.error(f"DocumentSearchTool.search Qdrant error: {e}", exc_info=True)
        else:
            logger.info("DocumentSearchTool.search: Qdrant unavailable, пропускаем семантический поиск")

        # TODO: если documents пуст либо нужно дополнить, выполнить полнотекстовый поиск в Postgres:
        # Например:
        # if not documents:
        #     async with AsyncSessionLocal() as session:
        #         # Пример простого ILIKE-поиска:
        #         stmt = select(DocumentRecord).where(DocumentRecord.content.ilike(f"%{query_text}%"))
        #         result = await session.execute(stmt.limit(req.top_k))
        #         recs = result.scalars().all()
        #         for rec in recs:
        #             snippet = rec.content[:200] + '...' if rec.content else ''
        #             documents.append(DocumentItem(id=rec.id, title=rec.title, content=snippet, metadata=rec.metadata or {}))
        #     logger.debug(f"DocumentSearchTool.search: Postgres fallback returned {len(documents)} docs")

        total = len(documents)
        return DocumentSearchResponse(documents=documents, total=total)

    async def index_document(self, doc_id: str, title: str, content: str, metadata: Dict[str, Any]):
        """
        Индексируем документ: сохраняем или обновляем в Postgres, 
        затем в Qdrant (если доступен).
        """
        # Сохраняем / обновляем Postgres
        try:
            async with AsyncSessionLocal() as session:
                record = await session.get(DocumentRecord, doc_id)
                if record:
                    record.title = title
                    record.content = content
                    record.meta = metadata
                else:
                    record = DocumentRecord(id=doc_id, title=title, content=content, meta=metadata)
                    session.add(record)
                await session.commit()
            logger.info(f"DocumentSearchTool: indexed in Postgres: {doc_id}")
        except Exception as e:
            logger.error(f"DocumentSearchTool.index_document Postgres error: {e}", exc_info=True)

        # Индекс в Qdrant
        if self.qdrant and self.embedder:
            try:
                embedding = await asyncio.to_thread(self.embedder.encode, content)
                # upsert в фоне
                await asyncio.to_thread(
                    self.qdrant.upsert,
                    collection_name=self.collection_name,
                    points=[rest.PointStruct(id=doc_id, vector=embedding.tolist(), payload=metadata)]
                )
                logger.info(f"DocumentSearchTool: indexed in Qdrant: {doc_id}")
            except Exception as e:
                logger.error(f"DocumentSearchTool.index_document Qdrant error: {e}", exc_info=True)
        else:
            logger.info("DocumentSearchTool.index_document: Qdrant unavailable, только Postgres")

    async def delete_document(self, doc_id: str):
        """
        Удаляем документ из Postgres и из Qdrant.
        """
        # Postgres
        try:
            async with AsyncSessionLocal() as session:
                record = await session.get(DocumentRecord, doc_id)
                if record:
                    await session.delete(record)
                    await session.commit()
                    logger.info(f"DocumentSearchTool: deleted from Postgres: {doc_id}")
                else:
                    logger.warning(f"DocumentSearchTool: delete_document: no such id in Postgres: {doc_id}")
        except Exception as e:
            logger.error(f"DocumentSearchTool.delete_document Postgres error: {e}", exc_info=True)

        # Qdrant
        if self.qdrant:
            try:
                await asyncio.to_thread(
                    self.qdrant.delete,
                    collection_name=self.collection_name,
                    points_selector=rest.PointIdsList([doc_id])
                )
                logger.info(f"DocumentSearchTool: deleted from Qdrant: {doc_id}")
            except Exception as e:
                logger.error(f"DocumentSearchTool.delete_document Qdrant error: {e}", exc_info=True)
        else:
            logger.info("DocumentSearchTool.delete_document: Qdrant unavailable")
