import json
import asyncio
from typing import Dict, Any, List, Optional
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from base_agent import BaseAgent, AgentContext, with_retry, with_timeout


class AnswerGeneratorAgent(BaseAgent):
    """
    Агент для генерации ответов на основе извлеченных документов и обогащенного контекста.
    Генерирует структурированные ответы с указанием источников и дополнительной информацией.
    """
    
    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("answer_generator", config, langfuse_client)
        
        # Инициализация LLM с параметрами из конфига
        self.llm = Ollama(
            model=self.get_config("model_name", "llama3.1:8b"),
            temperature=self.get_config("temperature", 0.3),
            base_url=self.get_config("ollama_base_url", "http://localhost:11434")
        )
        
        # Настройки агента
        self.top_k = self.get_config("top_k", 5)
        self.max_snippet_chars = self.get_config("max_snippet_chars", 500)
        self.include_sources = self.get_config("include_sources", True)
        
        # Шаблон промпта
        self.prompt_template = PromptTemplate(
            input_variables=["question", "doc_context", "enriched_context"],
            template=self._get_prompt_template()
        )

    def _get_prompt_template(self) -> str:
        """
        Возвращает шаблон промпта для генерации ответа.
        """
        return """
Ты - эксперт по предоставлению ответов на вопросы на основе корпоративной документации.

Вопрос: {question}

Используемая информация из документов (ссылки на источники указаны в квадратных скобках):

{doc_context}

Обогащенный контекст (дополнительная информация, связанные темы, уточняющие вопросы и т.д.):
{enriched_context}

Задача: на основе представленной информации сформировать полный, точный и понятный ответ на вопрос. 
Если возможно, указывай ссылки на документы в тексте ответа в формате [Источник N], где N соответствует порядку из списка документов выше. 
Если информация неполна, отметь это и предложи уточнение или дальнейшие шаги.

Ответь в формате JSON без лишнего текста:
{
  "answer_text": "...",
  "cited_sources": ["Источник 1", "Источник 2"],
  "follow_up_questions": ["..."],
  "used_documents": [ {"source": "...", "title": "..."} ]
}
"""

    @with_timeout(30.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        """
        Основная логика обработки запроса.
        """
        # Проверка наличия необходимых данных
        if not context.metadata or not context.metadata.get("retrieved_docs"):
            raise ValueError("Отсутствуют извлеченные документы для генерации ответа")
        
        # Получение и подготовка документов
        retrieved_docs = context.metadata["retrieved_docs"]
        sorted_docs = self._sort_documents(retrieved_docs)
        selected_docs = sorted_docs[:self.top_k]
        
        # Формирование контекста из документов
        doc_context, used_documents = self._create_doc_context(selected_docs)
        
        # Подготовка обогащенного контекста
        enriched_context = self._format_enriched_context(context.metadata.get("enriched_context"))
        
        # Формирование промпта и вызов LLM
        prompt = self.prompt_template.format(
            question=context.original_query,
            doc_context=doc_context,
            enriched_context=enriched_context
        )
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
        except Exception as e:
            self.logger.error(f"Ошибка при вызове LLM: {e}")
            raise
        
        # Парсинг ответа
        return self._process_llm_response(response, used_documents)

    def _sort_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Сортирует документы по оценке релевантности"""
        try:
            return sorted(docs, key=lambda x: x.get("score", 0), reverse=True)
        except Exception as e:
            self.logger.warning(f"Не удалось отсортировать документы: {e}")
            return docs

    def _create_doc_context(self, docs: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        """Создает контекст из документов и возвращает информацию об использованных документах"""
        doc_context_items = []
        used_documents = []
        
        for idx, doc in enumerate(docs, start=1):
            source = doc.get("source") or doc.get("id") or f"Документ {idx}"
            title = doc.get("title") or source
            content = doc.get("snippet") or doc.get("content", "")
            
            # Ограничение длины сниппета
            snippet = content.strip().replace("\n", " ")[:self.max_snippet_chars]
            
            doc_context_items.append(f"{idx}. {title}: {snippet} [Источник {idx}]")
            used_documents.append({"source": source, "title": title})
        
        return "\n".join(doc_context_items), used_documents

    def _format_enriched_context(self, enriched: Dict[str, Any]) -> str:
        """Форматирует обогащенный контекст в читаемую строку"""
        if not isinstance(enriched, dict):
            return "нет"
            
        enriched_parts = []
        
        # Дополнительная информация
        if ai := enriched.get("additional_information"):
            enriched_parts.append("Дополнительная информация: " + "; ".join(ai))
        
        # Связанные темы
        if rt := enriched.get("related_topics"):
            enriched_parts.append("Связанные темы: " + "; ".join(rt))
        
        # Уточняющие вопросы
        if fu := enriched.get("follow_up_questions"):
            enriched_parts.append("Уточняющие вопросы: " + "; ".join(fu))
        
        # Риски и ограничения
        if rl := enriched.get("risks_and_limitations"):
            enriched_parts.append("Риски и ограничения: " + "; ".join(rl))
        
        # Мета-информация
        if mi := enriched.get("meta_info"):
            meta_items = []
            if sources := mi.get("sources"):
                meta_items.append("Источники для чтения: " + "; ".join(sources))
            if formats := mi.get("formats"):
                meta_items.append("Рекомендуемые форматы: " + "; ".join(formats))
            if keywords := mi.get("keywords"):
                meta_items.append("Ключевые слова: " + "; ".join(keywords))
            
            if meta_items:
                enriched_parts.append("Мета-информация: " + ". ".join(meta_items))
        
        return "\n".join(enriched_parts) if enriched_parts else "нет"

    def _process_llm_response(self, response: str, used_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Обрабатывает ответ LLM и возвращает структурированный результат"""
        try:
            result = json.loads(response)
            result.setdefault("used_documents", used_documents)
            return result
        except json.JSONDecodeError:
            self.logger.warning("Не удалось распарсить JSON от LLM")
            return self._get_default_response(response, used_documents)

    def _get_default_response(self, raw_response: str, used_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Возвращает стандартный ответ при ошибке парсинга"""
        return {
            "answer_text": raw_response.strip(),
            "cited_sources": [],
            "follow_up_questions": [],
            "used_documents": used_documents
        }

    async def _postprocess(self, result_data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Дополнительная обработка результата"""
        # Здесь можно добавить логику фильтрации или кэширования
        return result_data

    def _calculate_confidence(self, result_data: Dict[str, Any], context: AgentContext) -> float:
        """Рассчитывает уровень уверенности на основе качества ответа"""
        if not isinstance(result_data, dict):
            return 0.0
            
        confidence = 0.5  # Базовая уверенность
        
        # Проверяем наличие основного ответа
        if result_data.get("answer_text"):
            confidence += 0.3
            
        # Проверяем наличие источников
        if result_data.get("cited_sources"):
            confidence += 0.2
            
        return min(1.0, confidence)  # Ограничиваем максимальное значение


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