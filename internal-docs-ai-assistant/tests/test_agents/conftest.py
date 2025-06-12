# файл tests/conftest.py
import pytest
import requests
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

@pytest.fixture(scope="session")
def ollama_health():
    """
    Проверяет, доступен ли Ollama LLM по адресу из переменной окружения или по умолчанию.
    Если недоступен, пропускает тесты, которые требуют LLM.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        resp = requests.get(base_url, timeout=2)
        if resp.status_code == 200:
            return base_url
        else:
            pytest.skip(f"Ollama health check returned status {resp.status_code}")
    except Exception:
        pytest.skip("Ollama server not available at " + base_url)

@pytest.fixture
def chorma_embedded(tmp_path):
    """
    Инициализирует embedded Chroma в временной директории, добавляет пару тестовых документов
    и возвращает объект vectorstore.
    """
    persist_dir = tmp_path / "chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="testcol"
    )
    # добавляем пару документов для тестов
    docs = [
        Document(page_content="Как оформить отпуск: пошаговая инструкция", metadata={"source": "doc1", "title": "Оформление отпуска"}),
        Document(page_content="Правила отпуска для сотрудников компании", metadata={"source": "doc2", "title": "Правила отпуска"}),
    ]
    vectorstore.add_documents(docs)
    # В новых версиях Chroma авто-сохранение, но вызываем persist на случай старых:
    try:
        vectorstore.persist()
    except Exception:
        pass
    return vectorstore
