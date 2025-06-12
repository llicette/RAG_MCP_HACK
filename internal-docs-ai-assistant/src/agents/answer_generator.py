import json
import asyncio
from typing import Dict, Any, List, Optional
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from agents.base_agent import BaseAgent, AgentContext, AgentResult, with_retry, with_timeout

class AnswerGeneratorAgent(BaseAgent):
    """Агент для генерации ответов на основе извлеченных документов и обогащенного контекста"""
    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("answer_generator", config, langfuse_client)
        # Настройка LLM
        self.llm = Ollama(
            model=self.get_config("model_name", "llama3.1:8b"),
            temperature=self.get_config("temperature", 0.3),
            base_url=self.get_config("ollama_base_url", "http://localhost:11434")
        )
        # Параметры конфигурации
        self.top_k = self.get_config("top_k", 5)
        self.max_snippet_chars = self.get_config("max_snippet_chars", 500)
        self.include_sources = self.get_config("include_sources", True)
        # Шаблон промпта: экранируем {{ и }} для JSON
        template = (
            "Ты - эксперт по предоставлению ответов на вопросы на основе корпоративной документации.\n"
            "Вопрос: {question}\n"
            "Используемая информация из документов (ссылки на источники указаны в квадратных скобках):\n"
            "{doc_context}\n"
            "Обогащенный контекст (дополнительная информация, связанные темы, уточняющие вопросы и т.д.):\n"
            "{enriched_context}\n"
            "Задача: на основе представленной информации сформировать полный, точный и понятный ответ на вопрос. "
            "Если возможно, указывай ссылки на документы в тексте ответа в формате [Источник N], где N соответствует порядку из списка документов выше. "
            "Если информация неполна, отметь это и предложи уточнение или дальнейшие шаги. "
            "Ответи в формате JSON без лишнего текста, структура: \n"
            "{{\"answer_text\": \"...\", "
            "\"cited_sources\": [\"Источник 1\", \"Источник 2\"], "
            "\"follow_up_questions\": [\"...\"], "
            "\"used_documents\": [ {{\"source\": \"...\", \"title\": \"...\"}} ]}}\n"
        )
        self.prompt_template = PromptTemplate(
            input_variables=["question", "doc_context", "enriched_context"],
            template=template
        )

    @with_timeout(30.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        question = context.original_query or ""
        # Получаем retrieved_docs
        retrieved = context.metadata.get("retrieved_docs")
        if not retrieved or not isinstance(retrieved, list):
            raise ValueError("Отсутствуют извлеченные документы для генерации ответа")
        # Сортируем по score, если есть
        try:
            sorted_docs = sorted(retrieved, key=lambda x: x.get("score", 0), reverse=True)
        except Exception:
            sorted_docs = retrieved
        selected_docs = sorted_docs[: self.top_k]
        doc_context_items = []
        used_documents: List[Dict[str, Any]] = []
        for idx, doc in enumerate(selected_docs, start=1):
            source = doc.get("source") or doc.get("id") or f"Документ {idx}"
            title = doc.get("title") or source
            content = doc.get("snippet") or doc.get("content", "")
            snippet = content.strip().replace("\n", " ")[: self.max_snippet_chars]
            doc_context_items.append(f"{idx}. {title}: {snippet} [Источник {idx}]")
            used_documents.append({"source": source, "title": title})
        doc_context = "\n".join(doc_context_items)
        enriched = context.metadata.get("enriched_context") or {}
        enriched_parts = []
        if isinstance(enriched, dict):
            ai = enriched.get("additional_information") or []
            if ai:
                enriched_parts.append("Дополнительная информация: " + "; ".join(ai))
            rt = enriched.get("related_topics") or []
            if rt:
                enriched_parts.append("Связанные темы: " + "; ".join(rt))
            fu = enriched.get("follow_up_questions") or []
            if fu:
                enriched_parts.append("Уточняющие вопросы: " + "; ".join(fu))
            rl = enriched.get("risks_and_limitations") or []
            if rl:
                enriched_parts.append("Риски и ограничения: " + "; ".join(rl))
            mi = enriched.get("meta_info") or {}
            if isinstance(mi, dict):
                sources = mi.get("sources") or []
                formats = mi.get("formats") or []
                keywords = mi.get("keywords") or []
                meta_items = []
                if sources:
                    meta_items.append("Источники для чтения: " + "; ".join(sources))
                if formats:
                    meta_items.append("Рекомендуемые форматы: " + "; ".join(formats))
                if keywords:
                    meta_items.append("Ключевые слова: " + "; ".join(keywords))
                if meta_items:
                    enriched_parts.append("Мета-информация: " + ". ".join(meta_items))
        enriched_context = "\n".join(enriched_parts) if enriched_parts else "нет"
        prompt = self.prompt_template.format(
            question=question,
            doc_context=doc_context,
            enriched_context=enriched_context
        )
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
        except Exception as e:
            self.logger.error(f"Ошибка при вызове LLM: {e}")
            raise
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning("Не удалось распарсить JSON от LLM, возвращаем текст ответа без структурированных полей")
            return {
                "answer_text": response.strip(),
                "cited_sources": [],
                "follow_up_questions": [],
                "used_documents": used_documents
            }
        result.setdefault("used_documents", used_documents)
        return result

    async def _postprocess(self, result_data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        return result_data

    def _calculate_confidence(self, result_data: Any, context: AgentContext) -> float:
        if isinstance(result_data, dict):
            answer = result_data.get("answer_text")
            if answer:
                return 0.8
        return 0.0



def create_answer_generator_agent(config: Dict[str, Any] = None, langfuse_client=None) -> AnswerGeneratorAgent:
    """
    Фабричная функция для создания экземпляра AnswerGeneratorAgent.
    """
    default_config = {
        "model_name": "llama3.1:8b",
        "temperature": 0.3,
        "ollama_base_url": "http://localhost:11434",
        "top_k": 5,
        "max_snippet_chars": 500,
        "include_sources": True
    }
    
    if config:
        default_config.update(config)
        
    return AnswerGeneratorAgent(default_config, langfuse_client)