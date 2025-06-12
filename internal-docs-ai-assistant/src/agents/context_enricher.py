import json
import asyncio
from typing import Dict, Any, List, Optional
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from base_agent import BaseAgent, AgentContext, with_retry, with_timeout


class ContextEnricherAgent(BaseAgent):
    """
    Агент для обогащения контекста ответа.
    Добавляет дополнительную информацию, связанные темы и мета-данные к ответу.
    """
    
    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("context_enricher", config, langfuse_client)
        
        # Инициализация LLM с параметрами из конфига
        self.llm = Ollama(
            model=self.get_config("model_name", "llama3.1:8b"),
            temperature=self.get_config("temperature", 0.2),
            base_url=self.get_config("ollama_base_url", "http://localhost:11434")
        )
        
        # Шаблон промпта для обогащения контекста
        self.enrich_prompt = PromptTemplate(
            input_variables=["question", "current_answer", "retrieved_info", "metadata"],
            template=self._get_enrich_prompt()
        )

    def _get_enrich_prompt(self) -> str:
        """
        Возвращает шаблон промпта для обогащения контекста.
        """
        return ("""
            Ты - эксперт по обогащению контекста для ответов на пользовательские запросы.

            Запрос: {question}

            Текущий ответ (если есть): {current_answer}

            Извлеченная информация / источники (если есть): {retrieved_info}

            Дополнительные метаданные (если есть): {metadata}

            Твоя задача:
            1. Добавить релевантную дополнительную информацию или фон, который поможет глубже раскрыть тему запроса.
            2. Предложить связанные темы или аспекты, которые могут быть интересны пользователю.
            3. Сформулировать уточняющие или follow-up вопросы, которые могут помочь уточнить запрос или углубить диалог.
            4. Выделить потенциальные риски или ограничения, связанные с темой (если применимо).
            5. Предложить мета-информацию: возможные источники для дальнейшего чтения, форматы представления (например, таблицы, графики), ключевые термины для поиска.

            Ответь в JSON формате без лишнего текста:
            {
            "additional_information": ["..."],
            "related_topics": ["..."],
            "follow_up_questions": ["..."],
            "risks_and_limitations": ["..."],
            "meta_info": {
                "sources": ["..."],
                "formats": ["..."],
                "keywords": ["..."]
            }
            }
            """)

    @with_timeout(25.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        """
        Основная логика обработки запроса.
        """
        # Извлечение данных из контекста
        question = context.original_query or ""
        metadata = context.metadata or {}
        
        # Получение текущего ответа
        current_answer = metadata.get("answer", "нет")
        
        # Обработка извлеченных документов
        retrieved_info = self._process_retrieved_docs(metadata.get("retrieved_docs"))
        
        # Получение метаданных
        metadata_info = str(metadata.get("additional_context", {}))
        
        # Формирование промпта
        prompt = self.enrich_prompt.format(
            question=question,
            current_answer=current_answer,
            retrieved_info=retrieved_info,
            metadata=metadata_info
        )
        
        # Вызов LLM
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
        except Exception as e:
            self.logger.error(f"Ошибка при вызове LLM: {e}")
            raise
        
        # Парсинг JSON-ответа
        return self._parse_llm_response(response)

    def _process_retrieved_docs(self, docs: Any) -> str:
        """
        Обрабатывает извлеченные документы в строковое представление.
        """
        if not isinstance(docs, list):
            return str(docs) if docs else "нет"
            
        snippets = []
        for doc in docs:
            title = doc.get("title") or doc.get("id") or ""
            content = doc.get("content", "")[:200]
            
            if title:
                snippets.append(f"{title}: {content}")
            else:
                snippets.append(content)
                
        return "; ".join(snippets) if snippets else "нет"

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Парсит ответ LLM и возвращает структурированный результат.
        """
        try:
            result = json.loads(response)
            return self._validate_response_structure(result)
        except json.JSONDecodeError:
            self.logger.warning("Не удалось распарсить JSON от LLM")
            return self._get_default_response()

    def _validate_response_structure(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверяет структуру ответа и заполняет отсутствующие поля.
        """
        default_response = self._get_default_response()
        
        # Проверяем каждый ключ
        for key in default_response:
            if key not in result:
                result[key] = default_response[key]
                
        return result

    def _get_default_response(self) -> Dict[str, Any]:
        """
        Возвращает стандартный ответ при ошибке.
        """
        return {
            "additional_information": [],
            "related_topics": [],
            "follow_up_questions": [],
            "risks_and_limitations": [],
            "meta_info": {
                "sources": [],
                "formats": [],
                "keywords": []
            }
        }

    async def _postprocess(self, result_data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """
        Дополнительная обработка результата.
        """
        # Здесь можно добавить логику фильтрации или кэширования
        return result_data

    def _calculate_confidence(self, result_data: Dict[str, Any], context: AgentContext) -> float:
        """
        Рассчитывает уровень уверенности на основе количества заполненных полей.
        """
        if not isinstance(result_data, dict):
            return 0.0
            
        # Проверяем ключевые разделы
        key_sections = [
            "additional_information", 
            "related_topics", 
            "follow_up_questions"
        ]
        
        filled_sections = sum(
            1 for section in key_sections 
            if isinstance(result_data.get(section), list) and result_data[section]
        )
        
        # Простая эвристика
        if filled_sections >= 3:
            return 0.9
        elif filled_sections > 0:
            return 0.5
        else:
            return 0.2


def create_context_enricher_agent(config: Dict[str, Any] = None, langfuse_client=None) -> ContextEnricherAgent:
    """
    Фабричная функция для создания экземпляра ContextEnricherAgent.
    """
    default_config = {
        "model_name": "llama3.1:8b",
        "temperature": 0.2,
        "ollama_base_url": "http://localhost:11434"
    }
    
    if config:
        default_config.update(config)
        
    return ContextEnricherAgent(default_config, langfuse_client)