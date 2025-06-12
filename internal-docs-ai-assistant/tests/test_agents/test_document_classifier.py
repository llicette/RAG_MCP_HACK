# tests/test_agents/test_document_classifier_real.py
import pytest
import os
import asyncio
from agents.document_classifier import DocumentClassifierAgent
from agents.base_agent import AgentContext

@pytest.mark.asyncio
async def test_document_classifier_simple_real(ollama_health):
    """
    Тестирование DocumentClassifierAgent с реальным LLM.
    Загружаем небольшой текст из тестового файла, вызываем агент и проверяем,
    что возвращается корректный JSON с ожидаемыми полями и допустимыми значениями.
    """
    base_url = ollama_health
    # Читаем тестовый текст (предполагаем, что это .txt файл с содержимым policy_sample)
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "test_data", "policy_sample.txt")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    agent = DocumentClassifierAgent(config={
        "model_name": "llama3.1:8b",
        "ollama_base_url": base_url,
    })
    # Подготавливаем context: document_info ожидается агентом
    context = AgentContext(
        user_id="user1",
        session_id="sess1",
        original_query="Классификация документа",
        metadata={
            "document_info": {
                "content": content,
                "filename": "policy_sample.txt"
            }
        }
    )
    result = await agent._process(context)
    # Проверяем, что вернулся объект с нужными ключами
    assert isinstance(result, dict)
    expected_keys = {
        "document_type", "priority", "target_audience",
        "confidence", "requires_updates", "confidential",
        "has_expiry", "estimated_validity_months",
        "key_topics", "document_purpose", "reasoning"
    }
    missing = expected_keys - result.keys()
    assert not missing, f"Отсутствуют поля: {missing}"

    # --- Проверяем document_type ---
    raw_doc_type = result["document_type"]
    # Если Enum, возьмём .value или .name.lower(), в зависимости от того, как определён Enum.
    if hasattr(raw_doc_type, "name"):
        doc_type_str = raw_doc_type.name.lower()
    elif hasattr(raw_doc_type, "value") and isinstance(raw_doc_type.value, str):
        doc_type_str = raw_doc_type.value
    else:
        doc_type_str = raw_doc_type  # ожидаем, что это уже строка
    assert isinstance(doc_type_str, str), f"document_type должен быть строкой, сейчас: {type(raw_doc_type)}"
    assert doc_type_str in {
        "policy", "procedure", "regulation", "form", "manual",
        "faq", "announcement", "report", "contract", "guide",
        "training", "technical", "other"
    }, f"Unexpected document_type: {doc_type_str}"

    # --- Проверяем priority ---
    raw_prio = result["priority"]
    # Если это Enum, извлечём имя и приведём к нижнему регистру
    if hasattr(raw_prio, "name"):
        prio_str = raw_prio.name.lower()
    elif hasattr(raw_prio, "value") and isinstance(raw_prio.value, str):
        prio_str = raw_prio.value
    else:
        prio_str = raw_prio  # если агент вернул уже строку
    assert isinstance(prio_str, str), f"priority должен быть строкой, сейчас: {type(raw_prio)}"
    assert prio_str in {"critical", "high", "medium", "low"}, f"Unexpected priority: {prio_str}"

    # --- target_audience: список строк или Enum-членов ---
    aud_list = result["target_audience"]
    assert isinstance(aud_list, list), "target_audience должно быть списком"
    allowed_aud = {
        "all_employees", "management", "hr", "it", "finance", "legal",
        "specific_role", "new_employees"
    }
    for item in aud_list:
        if hasattr(item, "name"):
            aud_str = item.name.lower()
        elif hasattr(item, "value") and isinstance(item.value, str):
            aud_str = item.value
        else:
            aud_str = item
        assert isinstance(aud_str, str), f"Элемент target_audience должен быть строкой, сейчас {type(item)}"
        assert aud_str in allowed_aud, f"Unexpected audience: {aud_str}"

    # --- confidence: число 0.0–1.0 ---
    conf = result["confidence"]
    assert isinstance(conf, (int, float)), f"confidence должно быть числом, сейчас {type(conf)}"
    assert 0.0 <= float(conf) <= 1.0, f"confidence вне диапазона [0,1]: {conf}"

    # --- булевы поля ---
    for bool_key in ["requires_updates", "confidential", "has_expiry"]:
        val = result[bool_key]
        assert isinstance(val, bool), f"{bool_key} должно быть bool, сейчас {type(val)}"

    # --- estimated_validity_months может быть None или число ---
    ev = result["estimated_validity_months"]
    assert (ev is None) or isinstance(ev, (int, float)), f"estimated_validity_months wrong type: {type(ev)}"

    # --- key_topics: список строк ---
    kt = result["key_topics"]
    assert isinstance(kt, list), "key_topics должно быть списком"
    for elem in kt:
        assert isinstance(elem, str), f"Элемент key_topics не строка: {elem}"

    # --- document_purpose, reasoning — строки ---
    assert isinstance(result["document_purpose"], str), "document_purpose должно быть строкой"
    assert isinstance(result["reasoning"], str), "reasoning должно быть строкой"
