import re
import asyncio
import time
from typing import Dict, Any, List, Optional
from collections import OrderedDict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, Qdrant
from langchain.schema import Document as LC_Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from base_agent import BaseAgent, AgentContext, AgentResult, with_retry, with_timeout

class RetrieverAgent(BaseAgent):
    """Агент для поиска и извлечения релевантной информации из векторной БД"""
    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("retriever", config, langfuse_client)
        # Настройка embedding
        model_name = self.get_config("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)
        # Параметры поиска
        self.top_k = self.get_config("top_k", 5)
        self.hybrid = self.get_config("hybrid", False)
        self.semantic_weight = self.get_config("semantic_weight", 0.7)
        self.keyword_weight = self.get_config("keyword_weight", 0.3)
        self.max_snippet_chars = self.get_config("max_snippet_chars", 500)
        self.with_metadata_filter = self.get_config("with_metadata_filter", False)
        # Метрики и кэш
        self.cache_size = self.get_config("cache_size", 100)
        self.cache: OrderedDict = OrderedDict()
        # Text splitter для индексирования
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # Инициализация векторного хранилища
        vs_type = self.get_config("vectorstore_type", "chroma").lower()
        if vs_type == "chroma":
            persist_dir = self.get_config("persist_directory")
            collection_name = self.get_config("collection_name", None)
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embedding,
                collection_name=collection_name
            )
        elif vs_type == "qdrant":
            url = self.get_config("qdrant_url")
            api_key = self.get_config("qdrant_api_key", None)
            collection_name = self.get_config("collection_name")
            prefer_grpc = self.get_config("qdrant_prefer_grpc", False)
            self.vectorstore = Qdrant(
                url=url,
                prefer_grpc=prefer_grpc,
                api_key=api_key,
                collection_name=collection_name,
                embedding=self.embedding
            )
        else:
            raise ValueError(f"Unsupported vectorstore_type: {vs_type}")

    def _cache_get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        if key in self.cache:
            # Обновляем порядок для LRU
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def _cache_set(self, key: str, value: List[Dict[str, Any]]):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.cache_size:
            # Удаляем старейший
            self.cache.popitem(last=False)

    @with_timeout(15.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> List[Dict[str, Any]]:
        query = context.processed_query or context.original_query
        if not query:
            raise ValueError("Нет текста запроса для поиска")
        # Ключ к кэшу: query + topic
        topic = context.topic or ""
        cache_key = f"{query}||{topic}".lower()
        cached = self._cache_get(cache_key)
        if cached is not None:
            self.logger.info(f"Результаты для запроса взяты из кэша: {query}")
            return cached
        # Подготовка фильтра по теме, если нужно
        filter_dict = None
        if self.with_metadata_filter and context.topic:
            filter_dict = {"topic": context.topic}
        # Семантический поиск
        start_time = time.time()
        try:
            sem_results = await asyncio.to_thread(
                self.vectorstore.similarity_search_with_score,
                query, k=self.top_k, filter=filter_dict
            )
        except TypeError:
            sem_results = await asyncio.to_thread(
                self.vectorstore.similarity_search_with_score,
                query, self.top_k, filter_dict
            )
        except Exception as e:
            self.logger.error(f"Ошибка при семантическом поиске: {e}")
            sem_results = []
        search_time = time.time() - start_time
        # Обработка результатов
        docs_scores = []
        for doc, score in sem_results:
            sim_score = float(score)
            docs_scores.append({"document": doc, "semantic_score": sim_score})
        # Гибридный поиск: лексический скор
        if self.hybrid and docs_scores:
            keywords = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
            for item in docs_scores:
                content = (item["document"].page_content or "").lower()
                count = sum(content.count(kw) for kw in keywords)
                lexical_score = count / (len(keywords) + 1)
                lexical_score = min(1.0, lexical_score)
                item["lexical_score"] = lexical_score
                item["score"] = self.semantic_weight * item["semantic_score"] + self.keyword_weight * lexical_score
            docs_scores.sort(key=lambda x: x["score"], reverse=True)
            docs_scores = docs_scores[: self.top_k]
        else:
            for item in docs_scores:
                item["score"] = item["semantic_score"]
        # Формирование списка результатов
        retrieved_docs: List[Dict[str, Any]] = []
        for idx, item in enumerate(docs_scores, start=1):
            doc: LC_Document = item["document"]
            score = item.get("score", 0.0)
            meta = doc.metadata or {}
            source = meta.get("source") or meta.get("id") or f"Документ_{idx}"
            title = meta.get("title") or source
            content = doc.page_content or ""
            snippet = content.strip().replace("\n", " ")
            if len(snippet) > self.max_snippet_chars:
                snippet = snippet[: self.max_snippet_chars].rsplit(" ", 1)[0] + "..."
            retrieved_docs.append({
                "source": source,
                "title": title,
                "content": snippet,
                "score": float(score)
            })
        # Сохраняем в metadata
        context.metadata["retrieved_docs"] = retrieved_docs
        # Логирование и метрики через LangFuse
        if self.langfuse:
            trace = self.langfuse.trace(
                name="retriever_search",
                user_id=context.user_id,
                session_id=context.session_id,
                input={"query": query, "topic": topic}
            )
            trace.update(
                output={"retrieved_count": len(retrieved_docs)},
                metadata={"search_time": search_time, "avg_score": (sum(item.get("score",0) for item in retrieved_docs)/len(retrieved_docs) if retrieved_docs else 0)}
            )
        self.logger.info(f"RetrieverAgent: найдено {len(retrieved_docs)} документов для запроса '{query}' за {search_time:.2f}s")
        # Кэширование
        self._cache_set(cache_key, retrieved_docs)
        return retrieved_docs

    async def index_documents(self, documents: List[Dict[str, Any]]):
        """Добавление новых документов в векторный индекс.
        documents: список словарей с полями: 'content': str, 'metadata': dict
        """
        chunks = []
        metadatas = []
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            splits = self.text_splitter.split_text(content)
            for i, chunk in enumerate(splits):
                chunk_meta = metadata.copy()
                if metadata.get("id"):
                    chunk_meta["source_id"] = metadata["id"]
                chunk_meta["chunk_index"] = i
                chunks.append(chunk)
                metadatas.append(chunk_meta)
        try:
            await asyncio.to_thread(
                self.vectorstore.add_texts,
                texts=chunks,
                metadatas=metadatas,
                ids=None
            )
            self.logger.info(f"Indexed {len(chunks)} chunks into vectorstore")
        except Exception as e:
            self.logger.error(f"Ошибка индексирования документов: {e}")

    def _calculate_confidence(self, result_data: Any, context: AgentContext) -> float:
        if isinstance(result_data, list) and result_data:
            avg = sum(item.get("score", 0.0) for item in result_data) / len(result_data)
            return float(min(1.0, avg))
        return 0.0

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
