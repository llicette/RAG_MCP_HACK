import logging
from typing import List, Dict, Any
from src.mcp.schemas.schemas import KnowledgeQueryRequest, KnowledgeQueryResponse, KnowledgeEntry
from src.mcp.db import AsyncSessionLocal, KnowledgeEntryRecord
from sqlalchemy import select
import uuid

logger = logging.getLogger(__name__)

class KnowledgeBaseTool:
    """
    Инструмент для работы с базой знаний: Postgres + SQLAlchemy
    """
    def __init__(self):
        pass

    async def query(self, req: KnowledgeQueryRequest) -> KnowledgeQueryResponse:
        logger.debug(f"KnowledgeBaseTool.query called with topic={req.topic}, context={req.context}")
        async with AsyncSessionLocal() as session:
            # Простая фильтрация: поиск в title или tags
            stmt = select(KnowledgeEntryRecord).where(
                KnowledgeEntryRecord.title.ilike(f"%{req.topic}%") |
                KnowledgeEntryRecord.summary.ilike(f"%{req.topic}%")
            )
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

    async def add_entry(self, entry: KnowledgeEntry) -> bool:
        async with AsyncSessionLocal() as session:
            rec = KnowledgeEntryRecord(
                id=entry.id or str(uuid.uuid4()),
                title=entry.title,
                summary=entry.summary,
                content=entry.content,
                tags=entry.tags
            )
            session.add(rec)
            await session.commit()
        logger.info(f"Knowledge entry added: {rec.id}")
        return True

    async def remove_entry(self, entry_id: str) -> bool:
        async with AsyncSessionLocal() as session:
            rec = await session.get(KnowledgeEntryRecord, entry_id)
            if rec:
                await session.delete(rec)
                await session.commit()
                logger.info(f"Knowledge entry removed: {entry_id}")
                return True
        return False