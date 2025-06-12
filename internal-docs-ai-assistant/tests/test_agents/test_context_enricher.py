# tests/test_agents/test_context_enricher_real.py
import pytest
import asyncio
from agents.context_enricher import ContextEnricherAgent
from agents.base_agent import AgentContext

@pytest.mark.asyncio
async def test_context_enricher_real(ollama_health):
    """
    Интеграционный тест ContextEnricherAgent с реальным Ollama.
    Проверяем, что возвращается корректный JSON с ожидаемыми ключами и типами полей.
    """
    base_url = ollama_health  # строка, например "http://localhost:11434"
    agent = ContextEnricherAgent(config={
        "model_name": "llama3.1:8b",
        "temperature": 0.2,
        "ollama_base_url": base_url
    })
    # Подготавливаем контекст: ответ уже есть, retrieved_docs приводим к небольшому списку
    # retrieved_docs может быть пустым или содержать примеры
    # Для интеграции добавим хотя бы один dummy-документ, чтобы LLM мог расширить контекст.
    # Но лучше real-case: пусть retrieved_docs пуст, LLM всё равно попытается дать доп.информацию.
    context = AgentContext(
        user_id="user1",
        session_id="sess1",
        original_query="Что такое CI/CD и зачем оно нужно?",
        metadata={
            "answer": "CI/CD — метод доставки ПО автоматически.",
            "retrieved_docs": []  # нет фактических документов, LLM выдаст общее расширение
        }
    )
    result = await agent._process(context)
    # Проверяем, что это dict с ожидаемыми ключами
    assert isinstance(result, dict)
    for key in ["additional_information", "related_topics", "follow_up_questions", "risks_and_limitations", "meta_info"]:
        assert key in result, f"Ожидали ключ {key} в результате"
    # Проверяем типы
    assert isinstance(result["additional_information"], list)
    assert isinstance(result["related_topics"], list)
    assert isinstance(result["follow_up_questions"], list)
    assert isinstance(result["risks_and_limitations"], list)
    assert isinstance(result["meta_info"], dict)
    # В meta_info ожидаем вложенные списки
    mi = result["meta_info"]
    for subkey in ["sources", "formats", "keywords"]:
        assert subkey in mi
        assert isinstance(mi[subkey], list)
