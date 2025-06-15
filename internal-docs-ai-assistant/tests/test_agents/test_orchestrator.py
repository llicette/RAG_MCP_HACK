# tests/test_agents/test_orchestrator.py  (фрагмент)

import pytest
import asyncio
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from agents.base_agent import AgentManager, AgentContext
from agents.question_critic import create_question_critic_agent
from agents.query_rewriter import create_query_rewriter_agent
from agents.topic_classifier import create_topic_classifier_agent
from agents.retriever_agent import create_retriever_agent  # фабрика
from agents.context_enricher import create_context_enricher_agent
from agents.answer_generator import create_answer_generator_agent
from agents.quality_checker import create_quality_checker_agent

@pytest.mark.asyncio
async def test_full_workflow_integration_real(ollama_health, tmp_path):
    """
    Интеграционный тест полного workflow: QuestionCritic -> QueryRewriter -> TopicClassifier ->
    Retriever (с embedded Chroma) -> ContextEnricher -> AnswerGenerator -> QualityChecker.
    Пропускается, если Ollama не доступна.
    """
    base_url = ollama_health

    # 1. Подготовка embedded Chroma и добавление документов
    persist_dir = tmp_path / "chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="wf_test"
    )
    docs = [
        Document(page_content="Процедура оформления отпуска: шаги и требования", metadata={"source": "doc1", "title": "Оформление отпуска"}),
        Document(page_content="Правила отпуска для сотрудников: условие и баланс дней", metadata={"source": "doc2", "title": "Правила отпуска"}),
    ]
    vectorstore.add_documents(docs)
    try:
        vectorstore.persist()
    except Exception:
        pass

    # 2. Регистрируем агентов
    manager = AgentManager(langfuse_client=None)
    # QuestionCritic
    manager.register_agent(create_question_critic_agent(config={
        "model_name": "llama3.1:8b",
        "ollama_base_url": base_url
    }))
    # QueryRewriter
    manager.register_agent(create_query_rewriter_agent(config={
        "model_name": "llama3.1:8b",
        "ollama_base_url": base_url
    }))
    # TopicClassifier
    manager.register_agent(create_topic_classifier_agent(config={
        "model_name": "llama3.1:8b",
        "ollama_base_url": base_url
    }))
    # Retriever: создать агент и инжектим vectorstore
    retr_agent = create_retriever_agent(config={"top_k": 2})
    setattr(retr_agent, "vectorstore", vectorstore)
    manager.register_agent(retr_agent)

    # ContextEnricher
    manager.register_agent(create_context_enricher_agent(config={
        "model_name": "llama3.1:8b",
        "ollama_base_url": base_url
    }))
    # AnswerGenerator
    manager.register_agent(create_answer_generator_agent(config={
        "model_name": "llama3.1:8b",
        "ollama_base_url": base_url
    }))
    # QualityChecker
    manager.register_agent(create_quality_checker_agent(config={
        "model_name": "llama3.1:8b",
        "ollama_base_url": base_url
    }))

    # 3. Запуск workflow через оркестратор
    # Предположим, у вас есть DocumentationOrchestrator:
    from core.orchestrator import DocumentationOrchestrator
    orchestrator = DocumentationOrchestrator(agent_manager=manager, config={})
    # Выполняем полный pipeline:
    user_id = "user1"
    session_id = "sess1"
    query = "Как оформить отпуск?"
    result = await orchestrator.process_query(user_id, session_id, query)
    # 4. Проверки результата
    assert result["success"] is True
    answer = result.get("answer")
    assert isinstance(answer, str) or isinstance(answer, dict)
    # Можно проверить, что в answer содержится какая-то рекомендация по отпуску:
    if isinstance(answer, str):
        assert "отпуск" in answer.lower()
    elif isinstance(answer, dict):
        # Если JSON-структура, ищем поле answer или answer_text
        text = answer.get("answer") or answer.get("answer_text") or ""
        assert "отпуск" in text.lower()

    # res = await orchestrator.process_query(user_id="u", session_id="s", query="Как оформить отпуск?")
    # # Проверяем, что workflow завершился без фатальной ошибки
    # assert isinstance(res, dict)
    # assert res.get("success") is True
    # # answer может быть None, если агенты не смогли сгенерировать корректный ответ.
    # # Но обычно ожидаем непустую строку. Проверяем, что ключ есть:
    # assert "answer" in res
    # # Если ответ получен, проверяем, что это строка. Если None, печатаем предупреждение, но не падаем?
    # # Дать гибкость: либо assert res["answer"] is not None, либо хотя бы str:
    # if res["answer"] is None:
    #     pytest.skip("AnswerGenerator не вернул ответ (возможно, промпт не дал результата).")
    # else:
    #     # если есть ответ, убеждаемся, что это непустая строка
    #     assert isinstance(res["answer"], str) and res["answer"].strip() != ""
    # # confidence должно быть число в диапазоне [0,1]
    # conf = res.get("confidence")
    # assert isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0
    # # В metadata гарантированно есть ключи confidence_scores и errors
    # metadata = res.get("metadata", {})
    # assert "confidence_scores" in metadata and isinstance(metadata["confidence_scores"], dict)
    # assert "errors" in metadata and isinstance(metadata["errors"], list)