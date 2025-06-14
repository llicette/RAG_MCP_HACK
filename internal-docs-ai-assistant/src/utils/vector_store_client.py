import os, asyncio, httpx
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from chromadb import Client as ChromaClient
from chromadb.config import Settings as ChromaSettings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from configs.settings import settings

class VectorStoreClient:
    def __init__(self):
        backend = settings.VECTOR_BACKEND.lower()
        self.backend = backend

        if backend == "chroma":
            embed_model = settings.EMBEDDING_MODEL_NAME
            self.embedder = SentenceTransformer(embed_model)
            if settings.CHROMA_SERVER_HOST:
                chroma_conf = ChromaSettings(
                    chroma_api_impl="rest",
                    chroma_server_host=settings.CHROMA_SERVER_HOST,
                    chroma_server_http_port=str(settings.CHROMA_SERVER_PORT),
                    persist_directory=settings.CHROMA_PERSIST_DIR
                )
                self.client = ChromaClient(client_settings=chroma_conf)
            else:
                self.client = ChromaClient(persist_directory=settings.CHROMA_PERSIST_DIR)
            self.collection = self.client.get_or_create_collection(name="documents",
                                                                   embedding_function=self.embedder.encode)

        elif backend == "qdrant":
            self.qdrant = QdrantClient(url=str(settings.QDRANT_URL))
            self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            # создаём коллекцию, если нужно
            try:
                self.qdrant.get_collection("documents")
            except:
                dim = self.embedder.get_sentence_embedding_dimension()
                self.qdrant.recreate_collection("documents",
                    vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE)
                )
        elif backend == "mcp":
            self.base_url = str(settings.MCP_BASE_URL)
            self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        else:
            raise ValueError(f"Unknown VECTOR_BACKEND: {backend}")

    async def search(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        filters = filters or {}
        if self.backend == "chroma":
            embedding = await asyncio.to_thread(self.embedder.encode, query)
            results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
            docs = []
            for hit, score in zip(results["documents"][0], results["distances"][0]):
                docs.append(dict(source=hit["id"], title=hit.get("title",""), content=hit["content"], score=score))
            return docs

        elif self.backend == "qdrant":
            embedding = await asyncio.to_thread(self.embedder.encode, query)
            hits = self.qdrant.search("documents", embedding.tolist(), limit=top_k, with_payload=True)
            docs = []
            for h in hits:
                payload = h.payload or {}
                docs.append(dict(source=str(h.id),
                                 title=payload.get("title",""),
                                 content=payload.get("snippet",""),
                                 metadata=payload,
                                 score=h.score))
            return docs

        else:  # mcp
            req = {"query": query, "top_k": top_k}
            async with httpx.AsyncClient(base_url=self.base_url, timeout=10) as client:
                r = await client.post("/document_search", json=req)
                r.raise_for_status()
                data = r.json()
            return data.get("documents", [])
