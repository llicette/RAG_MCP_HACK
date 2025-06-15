import asyncio
import logging
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent, AgentContext, with_retry, with_timeout

class RetrieverAgent(BaseAgent):
    """Агент для поиска документов через векторный стор (Chroma, Qdrant, MCP и т.д.)"""
    def __init__(self, config: Dict[str, Any], langfuse_client=None, vectorstore: Optional[Any] = None):
        """
        Если vectorstore передан явно, используем его напрямую.
        Иначе пытаемся инициализировать по конфигу (vectorstore_type), но с защитой от ошибок.
        """
        super().__init__("document_retriever", config, langfuse_client)
        self.logger = logging.getLogger(f"agent.{self.name}")
        # top_k
        try:
            self.top_k = int(self.get_config("top_k", config.get("top_k", 5)))
        except:
            self.top_k = 5

        if vectorstore is not None:
            # Инжектированный векторный стор (Chroma, etc.)
            self.vectorstore = vectorstore
            self._using_injected = True
            self.logger.info("RetrieverAgent: using injected vectorstore")
        else:
            self.vectorstore = None
            self._using_injected = False
            # Попытка инициализации по конфигу, но делаем это в try/except, чтобы не падать, если нет доступа
            backend = self.get_config("vectorstore_type", None)
            if backend is None:
                # можно прочитать из settings, например settings.VECTOR_BACKEND
                try:
                    from configs.settings import settings
                    backend = settings.VECTOR_BACKEND
                except Exception:
                    backend = None
            if backend == "chroma":
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    from langchain_community.vectorstores import Chroma
                    # Путь для персистенции
                    persist_dir = self.get_config("persist_directory", None) or getattr(settings, "CHROMA_PERSIST_DIR", None) or "chroma_db"
                    embeddings_model = self.get_config("embedding_model_name", None) or getattr(settings, "EMBEDDING_MODEL_NAME", None)
                    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
                    collection_name = self.get_config("collection_name", None)
                    if not collection_name:
                        import uuid
                        collection_name = f"col_{uuid.uuid4().hex}"
                    self.vectorstore = Chroma(
                        persist_directory=str(persist_dir),
                        embedding_function=embeddings,
                        collection_name=collection_name
                    )
                    self.logger.info(f"RetrieverAgent: initialized Chroma vectorstore at {persist_dir}, collection {collection_name}")
                except Exception as e:
                    self.logger.warning(f"RetrieverAgent: не удалось инициализировать Chroma: {e}")
                    self.vectorstore = None
            elif backend == "mcp":
                # Для MCP backend будем посылать HTTP-запросы внутри _process
                try:
                    from configs.settings import settings
                    import httpx
                    self.mcp_base_url = str(self.get_config("mcp_base_url", None) or settings.MCP_BASE_URL)
                    self.logger.info(f"RetrieverAgent: configured MCP backend at {self.mcp_base_url}")
                except Exception as e:
                    self.logger.warning(f"RetrieverAgent: не удалось настроить MCP backend: {e}")
                    self.mcp_base_url = None
                self.vectorstore = None
            elif backend == "qdrant":
                # Аналогично, попытка инициализировать Qdrant, но в тестах обычно не нужно
                try:
                    from qdrant_client import QdrantClient
                    from sentence_transformers import SentenceTransformer
                    import os
                    url = self.get_config("qdrant_url", None) or os.getenv("QDRANT_URL")
                    self.qdrant = QdrantClient(url=url)
                    model_name = self.get_config("embedding_model_name", None) or os.getenv("EMBEDDING_MODEL_NAME")
                    self.embedder = SentenceTransformer(model_name)
                    self.collection_name = self.get_config("collection_name", "documents")
                    self.logger.info(f"RetrieverAgent: initialized Qdrant at {url}, collection {self.collection_name}")
                    # Здесь можно оставить vectorstore=None и обрабатывать в _process вручную через self.qdrant
                except Exception as e:
                    self.logger.warning(f"RetrieverAgent: не удалось инициализировать Qdrant: {e}")
                    self.qdrant = None
                self.vectorstore = None
            else:
                self.logger.info(f"RetrieverAgent: неизвестный или не указан vectorstore_type ('{backend}'), не инициализируем стор автоматически")

    @with_timeout(60.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> List[Dict[str, Any]]:
        # Формируем query
        if context.processed_query and isinstance(context.processed_query, str) and context.processed_query.strip():
            query = context.processed_query.strip()
        else:
            query = (context.original_query or "").strip()
        if not query:
            self.logger.warning("RetrieverAgent: пустой запрос, возвращаем пустой список")
            return []

        # 1) Если есть injected vectorstore (Chroma и т.п.), используем его:
        if hasattr(self, "vectorstore") and self.vectorstore is not None:
            try:
                def sync_search():
                    # Попытка вызывать similarity_search_with_score или similarity_search
                    if hasattr(self.vectorstore, "similarity_search_with_score"):
                        return self.vectorstore.similarity_search_with_score(query, k=self.top_k)
                    elif hasattr(self.vectorstore, "similarity_search"):
                        docs = self.vectorstore.similarity_search(query, k=self.top_k)
                        return [(doc, None) for doc in docs]
                    else:
                        raise RuntimeError("injected vectorstore не поддерживает similarity_search")
                results = await asyncio.to_thread(sync_search)
                out = []
                for item in results:
                    if isinstance(item, tuple) and len(item) == 2:
                        doc, score = item
                    else:
                        doc = item
                        score = None
                    metadata = getattr(doc, "metadata", {}) or {}
                    source = metadata.get("source") or metadata.get("id") or None
                    title = metadata.get("title") or source or ""
                    content = getattr(doc, "page_content", None) or ""
                    out.append({
                        "source": source,
                        "title": title,
                        "content": content,
                        "score": score
                    })
                # Сортировка по score (если None, считаем 0)
                out = sorted(out, key=lambda x: x.get("score") or 0, reverse=True)
                return out
            except Exception as e:
                self.logger.error(f"RetrieverAgent: ошибка при вызове injected vectorstore: {e}", exc_info=True)
                return []

        # 2) MCP backend
        backend = self.get_config("vectorstore_type", None) or ""
        if backend == "mcp" and getattr(self, "mcp_base_url", None):
            try:
                from src.mcp.schemas.schemas import DocumentSearchRequest
                import httpx
                req = DocumentSearchRequest(query=query, top_k=self.top_k)
                async with httpx.AsyncClient(base_url=self.mcp_base_url) as client:
                    resp = await client.post('/document_search', json=req.dict(), headers={
                        'user_id': context.user_id or '', 'session_id': context.session_id or ''
                    })
                    resp.raise_for_status()
                    data = resp.json()
                out = []
                for doc in data.get("documents", []):
                    out.append({
                        "source": doc.get("id"),
                        "title": doc.get("title"),
                        "content": doc.get("content"),
                        "score": None
                    })
                return out
            except Exception as e:
                self.logger.error(f"RetrieverAgent: ошибка при вызове MCP: {e}", exc_info=True)
                return []

        # 3) Qdrant backend (без injected vectorstore) — при наличии self.qdrant
        if backend == "qdrant" and getattr(self, "qdrant", None):
            try:
                # Семпл-процесс: получить embedding и искать
                query_emb = await asyncio.to_thread(self.embedder.encode, query)
                hits = self.qdrant.search(
                    collection_name=self.collection_name,
                    query_vector=query_emb.tolist(),
                    limit=self.top_k,
                    with_payload=True
                )
                out = []
                import asyncio as _asyncio
                # Получаем содержимое из payload или Postgres: тут предполагаем простой payload
                for hit in hits:
                    source = hit.id
                    payload = hit.payload or {}
                    title = payload.get("title") or source
                    snippet = payload.get("snippet") or ""
                    out.append({
                        "source": source,
                        "title": title,
                        "content": snippet,
                        "score": hit.score if hasattr(hit, "score") else None
                    })
                return out
            except Exception as e:
                self.logger.error(f"RetrieverAgent: ошибка при вызове Qdrant: {e}", exc_info=True)
                return []

        # Если ничего не инициализировано
        self.logger.warning("RetrieverAgent: vectorstore не инициализирован, возвращаем пустой список")
        return []

def create_retriever_agent(config: Dict[str, Any] = None, langfuse_client=None, vectorstore: Optional[Any] = None) -> RetrieverAgent:
    default_config = {
        "vectorstore_type": "chroma",
        "top_k": 5,
    }
    if config:
        default_config.update(config)
    return RetrieverAgent(default_config, langfuse_client, vectorstore=vectorstore)
