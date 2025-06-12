# tests/test_agents/test_quality_checker_real.py
import pytest
import asyncio
from agents.quality_checker import QualityCheckerAgent
from agents.base_agent import AgentContext

@pytest.mark.asyncio
async def test_quality_checker_real(ollama_health):
    """
    Тест QualityCheckerAgent: даём ему некоторый ответ (процесс оформления отпуска),
    проверяем, что формат JSON корректен и overall_quality_score в диапазоне.
    """
    base_url = ollama_health
    agent = QualityCheckerAgent(config={
        "model_name": "llama3.1:8b",
        "temperature": 0.0,
        "ollama_base_url": base_url
    })
    # Подготовим контекст с запросом и примерным ответом
    question = "Как оформить отпуск?"
    # Можно для теста задать модельный ответ (короткий) или вызвать реальный AnswerGeneratorAgent,
    # но здесь вручную:
    answer = (
        "Чтобы оформить отпуск, отправьте заявку в HR через внутреннюю систему, "
        "укажите желаемые даты, дождитесь одобрения от руководителя. "
        "Проверьте баланс дней отпуска."
    )
    context = AgentContext(
        user_id="u",
        session_id="s",
        original_query=question,
        metadata={
            "answer": answer,
            "context_info": "Документы правил отпуска приложены"  # если есть
        }
    )
    result = await agent._process(context)
    assert isinstance(result, dict)
    # Ожидаемые ключи
    expected_keys = [
        "factual_accuracy_score",
        "completeness_score",
        "relevance_score",
        "clarity_score",
        "identified_issues",
        "suggestions",
        "overall_quality_score"
    ]
    for key in expected_keys:
        assert key in result, f"Ожидали ключ {key}"
    # Проверяем типы и диапазоны баллов
    for key in ["factual_accuracy_score","completeness_score","relevance_score","clarity_score","overall_quality_score"]:
        val = result[key]
        assert isinstance(val, (int,float)), f"{key} должно быть числом"
        assert 0.0 <= float(val) <= 1.0, f"{key} в диапазоне [0,1], получили {val}"
    # identified_issues и suggestions — списки строк
    assert isinstance(result["identified_issues"], list)
    assert isinstance(result["suggestions"], list)
