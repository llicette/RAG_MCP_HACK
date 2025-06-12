# tests/test_agents/test_answer_generator_real.py
import pytest
import asyncio
from agents.answer_generator import AnswerGeneratorAgent
from agents.base_agent import AgentContext
from langchain.schema import Document

@pytest.mark.asyncio
async def test_answer_generator_real(ollama_health, chorma_embedded):
    """
    Интеграционный тест AnswerGeneratorAgent: даём реальный retrieved_docs из chorma_embedded,
    вызываем LLM и проверяем, что ответ содержит нужные поля.
    """
    base_url = ollama_health
    # Предположим, у нас есть Chroma с теми же документами, что в предыдущих фикстурах
    # Получаем документы из vectorstore через similarity_search_with_score:
    query = "оформление отпуска"
    vs = chorma_embedded
    sem = vs.similarity_search_with_score(query, k=2)
    # Преобразуем в формат, ожидаемый AnswerGeneratorAgent:
    retrieved_docs = []
    for doc, score in sem:
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        retrieved_docs.append({
            "source": metadata.get("source") or "",
            "title": metadata.get("title") or "",
            "content": doc.page_content,
            "score": score
        })
    # Если вдруг retrieved_docs пуст, пропускаем тест
    if not retrieved_docs:
        pytest.skip("Нет retrieved_docs для AnswerGeneratorAgent")
    agent = AnswerGeneratorAgent(config={
        "model_name": "llama3.1:8b",
        "ollama_base_url": base_url
    })
    # Подготавливаем context. enriched_context можем дать пуст или скромный:
    context = AgentContext(
        user_id="u",
        session_id="s",
        original_query="Как оформить отпуск?",
        metadata={
            "retrieved_docs": retrieved_docs,
            "enriched_context": {
                # Если ContextEnricherAgent не запускать, можно дать пустые структуры:
                "additional_information": [],
                "related_topics": [],
                "follow_up_questions": [],
                "risks_and_limitations": [],
                "meta_info": {"sources": [], "formats": [], "keywords": []}
            }
        }
    )
    result = await agent._process(context)
    # Проверяем, что результат dict и содержит ключи answer_text, cited_sources, follow_up_questions, used_documents
    assert isinstance(result, dict)
    for key in ["answer_text", "cited_sources", "follow_up_questions", "used_documents"]:
        assert key in result, f"Ожидали ключ {key} в ответе"
    # Проверяем типы:
    assert isinstance(result["answer_text"], str) and result["answer_text"].strip() != ""
    assert isinstance(result["cited_sources"], list)
    assert isinstance(result["follow_up_questions"], list)
    assert isinstance(result["used_documents"], list)
