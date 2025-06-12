import pytest
import os
import asyncio
from agents.document_classifier import DocumentClassifierAgent
from agents.base_agent import AgentContext

@pytest.mark.asyncio
async def test_document_classifier_simple():
    # Читаем тестовый текст
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "test_data", "policy_sample.txt")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Создаём агент
    agent = DocumentClassifierAgent(config={"model_name": "llama3.1:8b", "ollama_base_url": "http://localhost:11434"})
    # Мокаем LLM - простой ответ
    dummy_response = '{"document_type": "policy", "priority": "medium", "target_audience": ["all_employees"], "confidence": 0.9, "requires_updates": false, "confidential": false, "has_expiry": false, "estimated_validity_months": null, "key_topics": ["политика", "ценности"], "document_purpose": "описание политики", "reasoning": "тест"}'
    class DummyLLM:
        def __init__(self, resp):
            self._resp = resp
        def invoke(self, prompt: str):
            return self._resp
    agent.llm = DummyLLM(dummy_response)
    # Подготавливаем context
    context = AgentContext(user_id="u1", session_id="s1", original_query="test", metadata={"document_info": {"content": content, "filename": "policy_sample.txt"}})
    result = await agent._process(context)
    # Проверяем поля
    assert result["document_type"].value == "policy"
    assert 0.0 <= result["confidence"] <= 1.0
    assert "priority" in result
