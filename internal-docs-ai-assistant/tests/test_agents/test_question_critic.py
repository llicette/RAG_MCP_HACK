# tests/test_agents/test_question_critic_real.py
import pytest
import asyncio
from agents.question_critic import QuestionCriticAgent
from agents.base_agent import AgentContext

@pytest.mark.asyncio
async def test_noise_query_real(ollama_health):
    """
    Проверяем, что QuestionCriticAgent определяет «шумовые» запросы, например короткое 'Привет'.
    """
    base_url = ollama_health
    agent = QuestionCriticAgent(config={
        "model_name": "llama3.1:8b",
        "ollama_base_url": base_url
    })
    context = AgentContext(user_id="u", session_id="s", original_query="Привет")
    result = await agent._process(context)
    assert isinstance(result, dict)
    # Ожидаем в структуре флаг is_noise_query
    assert "is_noise_query" in result
    assert isinstance(result["is_noise_query"], bool)
    # Если LLM логика срабатывает, для «Привет» это, скорее всего, is_noise_query=True
    # Но это зависит от реализации. Даем гибкость: просто проверяем поле есть.
    # Однако можно ожидать True:
    assert result["is_noise_query"] is True

@pytest.mark.asyncio
async def test_llm_analysis_real(ollama_health):
    """
    Проверяем, что QuestionCriticAgent с реальным LLM возвращает структуру анализа вопроса.
    Не проверяем точное число, но убеждаемся, что overall_score присутствует и в диапазоне 0-1.
    """
    base_url = ollama_health
    agent = QuestionCriticAgent(config={
        "model_name": "llama3.1:8b",
        "ollama_base_url": base_url,
        # Можно настроить таймаут больше, если LLM медленно отвечает:
        "timeout": 30.0
    })
    # Подготовим контекст с реально информативным вопросом
    question = "Как оформить отпуск сотруднику?"
    context = AgentContext(user_id="u", session_id="s", original_query=question)
    result = await agent._process(context)
    assert isinstance(result, dict)
    # Проверяем ключи анализа
    for key in ["clarity_score", "specificity_score", "grammar_score", "completeness_score", "context_score", "overall_score", "improved_question", "clarifying_questions", "topic_hints", "issues"]:
        assert key in result, f"Ожидаем ключ {key} в результате"
    # Проверяем диапазон баллов
    for score_key in ["clarity_score", "specificity_score", "grammar_score", "completeness_score", "context_score", "overall_score"]:
        val = result[score_key]
        assert isinstance(val, (int,float)), f"{score_key} должно быть числом"
        assert 0.0 <= float(val) <= 1.0, f"{score_key} должно быть в [0,1], получили {val}"
    # improved_question — строка, содержащая не пустой текст
    iq = result["improved_question"]
    assert isinstance(iq, str) and iq.strip() != ""
    # clarifying_questions — список строк (может быть пуст)
    assert isinstance(result["clarifying_questions"], list)
    # topic_hints — список строк (может быть пуст)
    assert isinstance(result["topic_hints"], list)
    # issues — список (может быть пуст)
    assert isinstance(result["issues"], list)
