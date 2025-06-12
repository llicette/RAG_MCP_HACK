import pytest
import asyncio
from agents.context_enricher import ContextEnricherAgent
from agents.base_agent import AgentContext

@pytest.mark.asyncio
async def test_context_enricher(monkeypatch):
    agent = ContextEnricherAgent(config={})
    # Мокаем LLM
    dummy_json = '{"additional_information": ["info1"], "related_topics": ["topic1"], "follow_up_questions": ["Как?"], "risks_and_limitations": ["risk1"], "meta_info": {"sources": ["src1"], "formats": ["fmt1"], "keywords": ["kw1"]}}'
    class DummyLLM:
        def __init__(self, resp): self._resp = resp
        def invoke(self, prompt): return self._resp
    agent.llm = DummyLLM(dummy_json)
    context = AgentContext(user_id="u", session_id="s", original_query="Вопрос", metadata={"answer": "Ответ", "retrieved_docs": []})
    result = await agent._process(context)
    assert "additional_information" in result
    assert result["additional_information"] == ["info1"]
