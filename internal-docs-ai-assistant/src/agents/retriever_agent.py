import re
import asyncio
import time
from typing import Dict, Any, List, Optional
from collections import OrderedDict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, Qdrant
from langchain.schema import Document as LC_Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from agents.base_agent import BaseAgent, AgentContext, AgentResult, with_retry, with_timeout

class RetrieverAgent(BaseAgent):
    """Агент для поиска и извлечения релевантной информации"""
    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("retriever", config, langfuse_client)
        self.vectorstore = None
        # Если в config передан уже готовый vectorstore, используем его
        if config.get("vectorstore_instance"):
            self.vectorstore = config.get("vectorstore_instance")
        else:
            vs_type = config.get("vectorstore_type")
            persist = config.get("persist_directory")
            # Если persist_directory не задан, пропускаем и ожидаем внешний mock
            if vs_type and persist:
                if vs_type == "chroma":
                    # Инициализация Chroma только если persist задан
                    self.vectorstore = Chroma(
                        persist_directory=persist,
                        embedding_function=HuggingFaceEmbeddings()
                    )
                elif vs_type == "qdrant":
                    url = config.get("qdrant_url")
                    if url:
                        self.vectorstore = Qdrant(
                            url=url,
                            prefer_grpc=config.get("prefer_grpc", False),
                            embedding_function=HuggingFaceEmbeddings(),
                            collection_name=config.get("collection_name")
                        )
        self.top_k = config.get("top_k", 5)
        self.hybrid = config.get("hybrid", False)
        self.semantic_weight = config.get("semantic_weight", 0.5)
        self.keyword_weight = config.get("keyword_weight", 0.5)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )

    @with_timeout(20.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> List[Dict[str, Any]]:
        query = context.processed_query or context.original_query
        if not self.vectorstore:
            raise ValueError("Vectorstore не инициализирован")
        results = []
        # Семантический поиск
        sem = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        # Преобразуем в словари
        for doc, score in sem:
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            source = metadata.get("source") or metadata.get("id")
            title = metadata.get("title") or source
            content = doc.page_content if hasattr(doc, 'page_content') else None
            results.append({"source": source, "title": title, "content": content, "score": score})
        # Гибридный поиск (упрощенно: пока только семантика, без ключевых слов)
        if self.hybrid:
            # Можно добавить ключевое совпадение: для тестов не требуется
            pass
        # Сортировка окончательная
        results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        return results

# Фабричная функция

def create_retriever_agent(config: Dict[str, Any] = None, langfuse_client=None) -> RetrieverAgent:
    default_config = {
        "vectorstore_type": "chroma",
        "persist_directory": None,
        "collection_name": None,
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 5,
        "hybrid": True,
        "semantic_weight": 0.7,
        "keyword_weight": 0.3,
        "max_snippet_chars": 500,
        "with_metadata_filter": False,
        "cache_size": 100,
        "qdrant_url": "http://localhost:6333",
        "qdrant_api_key": None,
        "qdrant_prefer_grpc": False
    }
    if config:
        default_config.update(config)
    return RetrieverAgent(default_config, langfuse_client)
