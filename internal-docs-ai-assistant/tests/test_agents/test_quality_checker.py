import pytest
import asyncio
from agents.base_agent import AgentContext
from agents.question_critic import QuestionCriticAgent
from agents.retriever_agent import RetrieverAgent
from agents.context_enricher import ContextEnricherAgent
from agents.answer_generator import AnswerGeneratorAgent
from agents.quality_checker import QualityCheckerAgent

@pytest.mark.asyncio
async def test_pipeline(monkeypatch):
    # Настраиваем агентов с dummy LLM и vectorstore
    qc = QuestionCriticAgent(config={})
    # Логика: вопрос проходит
    dummy_qc = '{"clarity_score": 0.8, "specificity_score": 0.8, "grammar_score": 0.9, "completeness_score": 0.8, "context_score": 0.5, "overall_score": 0.75, "issues": [], "improved_question": "Вопрос", "clarifying_questions": [], "topic_hints": ["HR"]}'
    class DummyLLM:
        def __init__(self, resp): self._resp = resp
        def invoke(self, prompt): return self._resp
    qc.llm = DummyLLM(dummy_qc)
    # Retriever с fake vectorstore
    retr = RetrieverAgent(config={"vectorstore_type": "chroma", "persist_directory": None, "top_k": 1})
    from langchain.schema import Document
    doc = Document(page_content="Test content", metadata={"source": "doc1", "title": "Doc1"})
    retr.vectorstore = type("FV", (), {"similarity_search_with_score": lambda self, q, k, filter=None: [(doc, 0.9)]})()
    # ContextEnricher
    ce = ContextEnricherAgent(config={})
    dummy_ce = '{"additional_information": [], "related_topics": [], "follow_up_questions": [], "risks_and_limitations": [], "meta_info": {"sources": [], "formats": [], "keywords": []}}'
    ce.llm = DummyLLM(dummy_ce)
    # AnswerGenerator
    ag = AnswerGeneratorAgent(config={})
    dummy_ag = '{"answer_text": "Ответ", "cited_sources": ["Источник 1"], "follow_up_questions": [], "used_documents": []}'
    ag.llm = DummyLLM(dummy_ag)
    # QualityChecker
    qc2 = QualityCheckerAgent(config={})
    dummy_qch = '{"factual_accuracy_score": 0.9, "completeness_score": 0.9, "relevance_score": 0.9, "clarity_score": 0.9, "identified_issues": [], "suggestions": [], "overall_quality_score": 0.9}'
    qc2.llm = DummyLLM(dummy_qch)

    # Контекст
    context = AgentContext(user_id="u", session_id="s", original_query="Как оформить отпуск?")

    # QuestionCritic
    res_qc = await qc._process(context)
    assert not res_qc["is_noise_query"]
    # Предположим, processed_query устанавливается равным improved_question
    context.processed_query = res_qc["improved_question"]
    context.topic = res_qc["topic_hints"][0] if res_qc["topic_hints"] else None

    # Retriever
    retrieved = await retr._process(context)
    assert retrieved

    # Добавляем retrieved_docs в metadata
    context.metadata["retrieved_docs"] = retrieved

    # ContextEnricher
    enriched = await ce._process(context)
    context.metadata["enriched_context"] = enriched

    # AnswerGenerator
    answer = await ag._process(context)
    context.metadata["answer"] = answer.get("answer_text")
    assert "answer_text" in answer

    # QualityChecker
    qc_res = await qc2._process(context)
    assert qc_res["overall_quality_score"] > 0.8
