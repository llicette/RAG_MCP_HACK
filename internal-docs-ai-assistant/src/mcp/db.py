import os
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Text, DateTime, Integer, JSON, func, Text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('POSTGRES_URL', 'postgresql+asyncpg://user:password@postgres:5432/ai_docs_db')

# Async engine и сессия
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Модель для хранения документов полного текста (для full-text поиска в Postgres)
class DocumentRecord(Base):
    __tablename__ = 'documents'
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    # tsvector column можно добавить через миграции: например, content_tsv tsvector GENERATED ALWAYS AS ...

# Модель для базы знаний
class KnowledgeEntryRecord(Base):
    __tablename__ = 'knowledge_entries'
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(JSON, default=[])
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

async def init_db():
    async with engine.begin() as conn:
        # Создаем таблицы при старте (в продакшне через миграции alembic)
        await conn.run_sync(Base.metadata.create_all)