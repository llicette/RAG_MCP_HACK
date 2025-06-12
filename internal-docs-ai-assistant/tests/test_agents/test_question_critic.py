import pytest
import asyncio
from agents.question_critic import QuestionCriticAgent
from agents.base_agent import AgentContext

@pytest.mark.asyncio
async def test_noise_query():
    agent = QuestionCriticAgent(config={})
    # Приветствие
    context = AgentContext(user_id="u", session_id="s", original_query="Привет")
    result = await agent._process(context)
    assert result["is_noise_query"]
    assert "suggested_response" in result

@pytest.mark.asyncio
async def test_llm_analysis(monkeypatch):
    agent = QuestionCriticAgent(config={})
    # Мокаем LLM
    dummy_json = '{"clarity_score": 0.8, "specificity_score": 0.7, "grammar_score": 0.9, "completeness_score": 0.8, "context_score": 0.6, "overall_score": 0.75, "issues": [], "improved_question": "Как оформить отпуск?", "clarifying_questions": [], "topic_hints": ["HR"]}'
    class DummyLLM:
        def __init__(self, resp): self._resp = resp
        def invoke(self, prompt): return self._resp
    agent.llm = DummyLLM(dummy_json)
    context = AgentContext(user_id="u", session_id="s", original_query="Как оформить отпуск?")
    result = await agent._process(context)
    assert not result["is_noise_query"]
    assert result["overall_score"] == pytest.approx(0.75)
    assert "improved_question" in result
