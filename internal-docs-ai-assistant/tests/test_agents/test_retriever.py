import pytest
import asyncio
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from agents.retriever_agent import RetrieverAgent
from agents.base_agent import AgentContext

@pytest.mark.asyncio
async def test_retriever_with_real_chroma(tmp_path):
    # 1. Инициализируем локальный Chroma в tmp_path
    persist_dir = tmp_path / "chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Задаём collection_name уникальное
    col_name = "testcol_" + tmp_path.name
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name=col_name
    )
    # 2. Добавляем документы
    docs = [
        Document(page_content="Как оформить отпуск: шаги и процесс", metadata={"source": "doc1", "title": "Отпуск"}),
        Document(page_content="Про больничный и ресурсы для сотрудников", metadata={"source": "doc2", "title": "Больничный"})
    ]
    vectorstore.add_documents(docs)
    vectorstore.persist()
    # 3. Создаём агент, передаём vectorstore напрямую
    agent = RetrieverAgent(config={"top_k": 2}, vectorstore=vectorstore)
    # 4. Вызываем процесс
    context = AgentContext(user_id="u", session_id="s", original_query="оформление отпуска")
    results = await agent._process(context)
    # 5. Ожидаем, что первый результат – doc1
    assert len(results) >= 1
    sources = [r["source"] for r in results]
    assert "doc1" in sources
