# src/mcp/tools/knowledge_base.py
import logging
import uuid
from typing import List
from sqlalchemy import select, or_
from mcp.schemas.schemas import KnowledgeQueryRequest, KnowledgeQueryResponse, KnowledgeEntry
from mcp.db import AsyncSessionLocal, KnowledgeEntryRecord

logger = logging.getLogger(__name__)

class KnowledgeBaseTool:
    """
    Инструмент для работы с базой знаний: Postgres + SQLAlchemy.
    """
    def __init__(self):
        # Если понадобятся какие-то инициализации, добавить здесь.
        pass

    async def query(self, req: KnowledgeQueryRequest) -> KnowledgeQueryResponse:
        logger.debug(f"KnowledgeBaseTool.query called with topic={req.topic}, context={getattr(req, 'context', None)}")
        try:
            async with AsyncSessionLocal() as session:
                # Защита от None или пустого topic
                topic = (req.topic or "").strip()
                if not topic:
                    # Если пустой запрос, вернуть 0 результатов
                    return KnowledgeQueryResponse(entries=[], total=0)

                # Экранирование %, _ при необходимости:
                safe = topic.replace("%", "\\%").replace("_", "\\_")
                stmt = select(KnowledgeEntryRecord).where(
                    or_(
                        KnowledgeEntryRecord.title.ilike(f"%{safe}%", escape="\\"),
                        KnowledgeEntryRecord.summary.ilike(f"%{safe}%", escape="\\")
                    )
                )
                # Если нужны пагинация:
                if hasattr(req, 'limit') and req.limit:
                    stmt = stmt.limit(req.limit)
                if hasattr(req, 'offset') and req.offset:
                    stmt = stmt.offset(req.offset)

                result = await session.execute(stmt)
                records = result.scalars().all()

                entries: List[KnowledgeEntry] = []
                for rec in records:
                    entries.append(KnowledgeEntry(
                        id=rec.id,
                        title=rec.title,
                        summary=rec.summary,
                        content=rec.content,
                        tags=rec.tags
                    ))
                total = len(entries)
            logger.debug(f"KnowledgeBaseTool.query returning {total} entries")
            return KnowledgeQueryResponse(entries=entries, total=total)
        except Exception as e:
            logger.error(f"KnowledgeBaseTool.query error: {e}", exc_info=True)
            # Можно поднять HTTPException или собственное исключение, пусть вызывающий код обрабатывает
            raise

    async def add_entry(self, entry: KnowledgeEntry) -> bool:
        try:
            async with AsyncSessionLocal() as session:
                new_id = entry.id or str(uuid.uuid4())
                rec = await session.get(KnowledgeEntryRecord, new_id)
                if rec:
                    # Обновление существующей записи:
                    rec.title = entry.title
                    rec.summary = entry.summary
                    rec.content = entry.content
                    rec.tags = entry.tags
                else:
                    rec = KnowledgeEntryRecord(
                        id=new_id,
                        title=entry.title,
                        summary=entry.summary,
                        content=entry.content,
                        tags=entry.tags
                    )
                    session.add(rec)
                await session.commit()
            logger.info(f"Knowledge entry added/updated: {new_id}")
            return True
        except Exception as e:
            logger.error(f"KnowledgeBaseTool.add_entry error: {e}", exc_info=True)
            raise

    async def remove_entry(self, entry_id: str) -> bool:
        try:
            async with AsyncSessionLocal() as session:
                rec = await session.get(KnowledgeEntryRecord, entry_id)
                if rec:
                    await session.delete(rec)
                    await session.commit()
                    logger.info(f"Knowledge entry removed: {entry_id}")
                    return True
                else:
                    logger.warning(f"Knowledge entry to remove not found: {entry_id}")
                    return False
        except Exception as e:
            logger.error(f"KnowledgeBaseTool.remove_entry error: {e}", exc_info=True)
            raise
