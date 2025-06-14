import pytest
import asyncio
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from agents.retriever_agent import RetrieverAgent, create_retriever_agent
from agents.base_agent import AgentContext

@pytest.mark.asyncio
async def test_retriever_with_real_chroma(tmp_path):
    # 1. Initialize a local Chroma DB in a temporary directory
    persist_dir = tmp_path / "chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    col_name = f"testcol_{tmp_path.name}"
    
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name=col_name
    )

    # 2. Add test documents
    docs = [
        Document(page_content="Как оформить отпуск: шаги и процесс", metadata={"source": "doc1", "title": "Отпуск"}),
        Document(page_content="Про больничный и ресурсы для сотрудников", metadata={"source": "doc2", "title": "Больничный"})
    ]
    vectorstore.add_documents(docs)
    vectorstore.persist()

    # 3. Create agent with test vectorstore
    agent = create_retriever_agent(config={"top_k": 2}, vectorstore=vectorstore)

    # 4. Run the agent
    context = AgentContext(user_id="u", session_id="s", original_query="оформление отпуска")
    results = await agent._process(context)

    # 5. Assertions
    assert len(results) >= 1
    sources = [r["source"] for r in results]
    assert "doc1" in sources