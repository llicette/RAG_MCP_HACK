# src/utils/vector_store_client.py

import logging
import os
import asyncio
from typing import List, Dict, Any

class VectorStoreClient:
    """
    Утилитный класс для инициализации нужного векторного стора по настройкам.
    Например, Qdrant, Chroma или MCP.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger("VectorStoreClient")
        self.config = config or {}
        from configs.settings import settings
        backend = self.config.get("vectorstore_type") or settings.VECTOR_BACKEND
        self.backend = backend
        self.client = None

        if backend == "chroma":
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from langchain_community.vectorstores import Chroma
                persist_dir = self.config.get("persist_directory") or settings.CHROMA_PERSIST_DIR or "chroma_db"
                embedding_model = self.config.get("embedding_model_name") or settings.EMBEDDING_MODEL_NAME
                embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
                collection_name = self.config.get("collection_name")
                if not collection_name:
                    import uuid
                    collection_name = f"col_{uuid.uuid4().hex}"
                self.client = Chroma(
                    persist_directory=str(persist_dir),
                    embedding_function=embeddings,
                    collection_name=collection_name
                )
                self.logger.info(f"VectorStoreClient: Chroma initialized at {persist_dir}, collection {collection_name}")
            except Exception as e:
                self.logger.warning(f"VectorStoreClient: failed to init Chroma: {e}")
                self.client = None

        elif backend == "qdrant":
            try:
                from qdrant_client import QdrantClient
                from sentence_transformers import SentenceTransformer
                QDRANT_URL = os.getenv('QDRANT_URL')
                QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', None)
                url = self.config.get("qdrant_url") or os.getenv("QDRANT_URL")
                self.qdrant = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    prefer_grpc=False
                )
                embedding_model = self.config.get("embedding_model_name") or os.getenv("EMBEDDING_MODEL_NAME")
                self.embedder = SentenceTransformer(embedding_model)
                self.collection_name = self.config.get("collection_name", "documents")
                # Можно проверить существование коллекции, но необязательно
                self.client = self  # сами используем поля qdrant/embedder
                self.logger.info(f"VectorStoreClient: Qdrant initialized at {url}")
            except Exception as e:
                self.logger.warning(f"VectorStoreClient: failed to init Qdrant: {e}")
                self.client = None

        elif backend == "mcp":
            try:
                import httpx
                self.mcp_base_url = str(self.config.get("mcp_base_url") or settings.MCP_BASE_URL)
                self.client = self  # используем в search
                self.logger.info(f"VectorStoreClient: MCP configured at {self.mcp_base_url}")
            except Exception as e:
                self.logger.warning(f"VectorStoreClient: failed to init MCP client: {e}")
                self.client = None
        else:
            self.logger.warning(f"VectorStoreClient: unknown backend '{backend}'")
            self.client = None

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.client:
            self.logger.warning("VectorStoreClient.search: client not initialized, returning []")
            return []
        if self.backend == "chroma":
            try:
                def sync_search():
                    if hasattr(self.client, "similarity_search_with_score"):
                        return self.client.similarity_search_with_score(query, k=top_k)
                    elif hasattr(self.client, "similarity_search"):
                        docs = self.client.similarity_search(query, k=top_k)
                        return [(doc, None) for doc in docs]
                    else:
                        raise RuntimeError("Chroma client has no similarity_search")
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
                    out.append({"source": source, "title": title, "content": content, "score": score})
                return sorted(out, key=lambda x: x.get("score") or 0, reverse=True)
            except Exception as e:
                self.logger.error(f"VectorStoreClient.search (Chroma) error: {e}", exc_info=True)
                return []

        elif self.backend == "qdrant":
            try:
                # Получаем embedding
                query_emb = await asyncio.to_thread(self.embedder.encode, query)
                hits = self.qdrant.search(
                    collection_name=self.collection_name,
                    query_vector=query_emb.tolist(),
                    limit=top_k,
                    with_payload=True
                )
                out = []
                for hit in hits:
                    source = hit.id
                    payload = hit.payload or {}
                    title = payload.get("title") or source
                    snippet = payload.get("snippet") or ""
                    out.append({"source": source, "title": title, "content": snippet, "score": getattr(hit, "score", None)})
                return out
            except Exception as e:
                self.logger.error(f"VectorStoreClient.search (Qdrant) error: {e}", exc_info=True)
                return []

        elif self.backend == "mcp":
            try:
                from src.mcp.schemas.schemas import DocumentSearchRequest
                import httpx
                req = DocumentSearchRequest(query=query, top_k=top_k)
                async with httpx.AsyncClient(base_url=self.mcp_base_url) as client:
                    resp = await client.post('/document_search', json=req.dict())
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
                self.logger.error(f"VectorStoreClient.search (MCP) error: {e}", exc_info=True)
                return []
        else:
            return []

# В коде RetrieverAgent можно, если хочется, делать:
# client = VectorStoreClient(config)
# и хранить client.client или client, 
# но для тестов достаточно инъекции Chroma напрямую в RetrieverAgent.
