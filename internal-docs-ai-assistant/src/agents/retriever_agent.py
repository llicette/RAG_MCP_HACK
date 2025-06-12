# src/agents/retriever_agent.py

import asyncio
import uuid
from typing import Dict, Any, List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document as LC_Document
from agents.base_agent import BaseAgent, AgentContext, with_retry, with_timeout


class RetrieverAgent(BaseAgent):
    """Агент для поиска и извлечения релевантной информации."""
    def __init__(self, config: Dict[str, Any], langfuse_client=None, vectorstore=None):
        super().__init__("document_retriever", config, langfuse_client)
        # Dependency Injection: если vectorstore передан извне, используем его
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            # Конфигурация Chroma
            persist_dir = config.get("persist_directory")
            # Если None или пусто, задаём значение по умолчанию
            if not persist_dir:
                persist_dir = "chroma_db"
            collection_name = config.get("collection_name")
            if not collection_name:
                collection_name = f"col_{uuid.uuid4().hex}"

            embed_model = config.get("embedding_model_name") or config.get("embed_model") or "sentence-transformers/all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbeddings(model_name=embed_model)

            # Если Chroma-сервер через REST:
            if config.get("chroma_server_host"):
                try:
                    from chromadb.config import Settings
                    settings = Settings(
                        chroma_api_impl="rest",
                        chroma_server_host=config["chroma_server_host"],
                        chroma_server_http_port=str(config["chroma_server_http_port"]),
                        persist_directory=persist_dir
                    )
                    self.vectorstore = Chroma(
                        client_settings=settings,
                        embedding_function=embeddings,
                        collection_name=collection_name
                    )
                except ImportError:
                    raise RuntimeError("chromadb.config.Settings нужен для подключения к Chroma REST API, но не найден")
            else:
                # Локальный Chroma с файловой персистенцией
                self.vectorstore = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )

        # Остальные параметры
        self.top_k = self.get_config("top_k", 5)
        self.hybrid = self.get_config("hybrid", False)
        self.semantic_weight = self.get_config("semantic_weight", 0.5)
        self.keyword_weight = self.get_config("keyword_weight", 0.5)

    @with_timeout(60.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> List[Dict[str, Any]]:
        query = context.processed_query or context.original_query or ""
        if not self.vectorstore:
            raise ValueError("Vectorstore не инициализирован")

        # Семантический поиск: similarity_search_with_score может быть синхронным
        # оборачиваем в to_thread, если блокирует:
        results = await asyncio.to_thread(lambda: self.vectorstore.similarity_search_with_score(query, k=self.top_k))

        docs: List[Dict[str, Any]] = []
        for doc, score in results:
            metadata = getattr(doc, "metadata", {}) or {}
            source = metadata.get("source") or metadata.get("id") or None
            title = metadata.get("title") or source or ""
            content = getattr(doc, "page_content", None)
            docs.append({
                "source": source,
                "title": title,
                "content": content,
                "score": score
            })

        # Гибридный поиск: при необходимости можно расширить
        if self.hybrid:
            # Здесь можно добавить ключевую фильтрацию или отдельный keyword-based search
            pass

        # Сортировка по score
        docs = sorted(docs, key=lambda x: x.get("score", 0), reverse=True)
        return docs

def create_retriever_agent(config: Dict[str, Any] = None, langfuse_client=None) -> RetrieverAgent:
    default_config = {
        "vectorstore_type": "chroma",
        "persist_directory": None,
        "collection_name": None,
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 5,
        "hybrid": False,
        "semantic_weight": 0.7,
        "keyword_weight": 0.3,
        "chroma_server_host": None,
        "chroma_server_http_port": None,
    }
    if config:
        default_config.update(config)
    return RetrieverAgent(default_config, langfuse_client)
