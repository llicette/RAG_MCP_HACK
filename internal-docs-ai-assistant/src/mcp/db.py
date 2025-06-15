# src/mcp/db.py
import os
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Text, DateTime, JSON, func
from configs.settings import settings

# DATABASE_URL берём из settings
DATABASE_URL = str(settings.POSTGRES_URL)  # ожидаем формат postgresql+asyncpg://...

# Async engine и сессия
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class DocumentRecord(Base):
    __tablename__ = 'documents'
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    meta = Column("metadata", JSON, default={})  # JSON для метаданных
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    # Для полнотекстового поиска можно добавить content_tsvector через миграции

class KnowledgeEntryRecord(Base):
    __tablename__ = 'knowledge_entries'
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(JSON, default=list)  # default=list для нового списка
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

async def init_db():
    """
    При старте приложению: создаём таблицы. 
    Для продакшн рекомендуется Alembic для миграций.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
