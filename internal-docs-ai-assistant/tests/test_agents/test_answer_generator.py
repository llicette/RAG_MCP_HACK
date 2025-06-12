import pytest
import asyncio
from agents.answer_generator import AnswerGeneratorAgent
from agents.base_agent import AgentContext

@pytest.mark.asyncio
async def test_answer_generator(monkeypatch):
    agent = AnswerGeneratorAgent(config={})
    # Мокаем retrieved_docs и enriched_context
    context = AgentContext(user_id="u", session_id="s", original_query="Вопрос", metadata={
        "retrieved_docs": [
            {"source": "doc1", "title": "Doc1", "content": "Содержимое 1", "score": 0.9},
            {"source": "doc2", "title": "Doc2", "content": "Содержимое 2", "score": 0.8},
        ],
        "enriched_context": {
            "additional_information": ["inf1"],
            "related_topics": ["top1"],
            "follow_up_questions": ["Как?"],
            "risks_and_limitations": ["risk"],
            "meta_info": {"sources": ["s1"], "formats": ["f1"], "keywords": ["k1"]}
        }
    })
    # Мокаем LLM
    dummy_json = '{"answer_text": "Ответ с источником [Источник 1]", "cited_sources": ["Источник 1"], "follow_up_questions": [], "used_documents": []}'
    class DummyLLM:
        def __init__(self, resp): self._resp = resp
        def invoke(self, prompt): return self._resp
    agent.llm = DummyLLM(dummy_json)
    result = await agent._process(context)
    assert "answer_text" in result
    assert "cited_sources" in result
    assert result["cited_sources"] == ["Источник 1"]
