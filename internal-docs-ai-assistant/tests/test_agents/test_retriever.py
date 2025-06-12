import pytest
import asyncio
from agents.retriever_agent import RetrieverAgent
from agents.base_agent import AgentContext
from langchain.schema import Document

@pytest.mark.asyncio
async def test_retriever_semantic_only(monkeypatch):
    # Создаём агент с dummy vectorstore
    agent = RetrieverAgent(config={"vectorstore_type": "chroma", "persist_directory": None, "top_k": 2})
    # Мокаем vectorstore
    doc1 = Document(page_content="Test content one", metadata={"source": "doc1", "title": "Doc1"})
    doc2 = Document(page_content="Another test", metadata={"source": "doc2", "title": "Doc2"})
    fake_results = [(doc1, 0.9), (doc2, 0.8)]
    class FakeVS:
        def similarity_search_with_score(self, query, k, filter=None):
            return fake_results[:k]
    agent.vectorstore = FakeVS()
    context = AgentContext(user_id="u", session_id="s", original_query="test query")
    results = await agent._process(context)
    assert len(results) == 2
    assert results[0]["source"] == "doc1"
    assert results[1]["source"] == "doc2"
    assert "score" in results[0]

@pytest.mark.asyncio
async def test_retriever_hybrid(monkeypatch):
    agent = RetrieverAgent(config={"vectorstore_type": "chroma", "persist_directory": None, "top_k": 2, "hybrid": True, "semantic_weight": 0.5, "keyword_weight": 0.5})
    # Fake vectorstore returns:
    from langchain.schema import Document
    doc1 = Document(page_content="Alpha beta gamma", metadata={"source": "doc1", "title": "Doc1"})
    doc2 = Document(page_content="Delta epsilon", metadata={"source": "doc2", "title": "Doc2"})
    fake_results = [(doc1, 0.7), (doc2, 0.6)]
    class FakeVS:
        def similarity_search_with_score(self, query, k, filter=None):
            return fake_results[:k]
    agent.vectorstore = FakeVS()
    context = AgentContext(user_id="u", session_id="s", original_query="alpha gamma")
    results = await agent._process(context)
    # Ожидаем, что doc1 (с двумя ключевыми словами) будет выше
    assert results[0]["source"] == "doc1"

