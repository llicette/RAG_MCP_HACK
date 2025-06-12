import json
import asyncio
from typing import Dict, Any
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from agents.base_agent import BaseAgent, AgentContext, with_retry, with_timeout

class ContextEnricherAgent(BaseAgent):
    """Агент для обогащения контекста ответа."""
    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("context_enricher", config, langfuse_client)
        self.llm = Ollama(
            model=self.get_config("model_name", "llama3.1:8b"),
            temperature=self.get_config("temperature", 0.2),
            base_url=self.get_config("ollama_base_url", "http://localhost:11434")
        )
        # Шаблон промпта с экранированием фигурных скобок для JSON-примера
        template = (
            "Ты - эксперт по обогащению контекста для ответов на пользовательские запросы.\n"
            "Запрос: {question}\n"
            "Текущий ответ: {current_answer}\n"
            "Извлеченная информация / источники: {retrieved_info}\n"
            "Дополнительные метаданные: {metadata}\n"
            "Задача:\n"
            "1. Добавить релевантную дополнительную информацию или фон, который поможет глубже раскрыть тему запроса.\n"
            "2. Предложить связанные темы или аспекты, которые могут быть интересны пользователю.\n"
            "3. Сформулировать уточняющие или follow-up вопросы, которые могут помочь уточнить запрос или углубить диалог.\n"
            "4. Выделить потенциальные риски или ограничения, связанные с темой (если применимо).\n"
            "5. Предложить мета-информацию: возможные источники для дальнейшего чтения, форматы представления, ключевые термины для поиска.\n"
            "Ответь в JSON формате без лишнего текста, структура:\n"
            "{{\"additional_information\": [\"...\"], \"related_topics\": [\"...\"], \"follow_up_questions\": [\"...\"], \"risks_and_limitations\": [\"...\"], \"meta_info\": {{\"sources\": [\"...\"], \"formats\": [\"...\"], \"keywords\": [\"...\"]}}}}"
        )
        self.enrich_prompt = PromptTemplate(
            input_variables=["question", "current_answer", "retrieved_info", "metadata"],
            template=template
        )

    @with_timeout(60.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        question = context.original_query or ""
        metadata = context.metadata or {}
        current_answer = metadata.get("answer", "нет")
        retrieved_info = self._process_retrieved_docs(metadata.get("retrieved_docs"))
        metadata_info = str(metadata.get("additional_context", {}))
        prompt = self.enrich_prompt.format(
            question=question,
            current_answer=current_answer,
            retrieved_info=retrieved_info,
            metadata=metadata_info
        )
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
        except Exception as e:
            self.logger.error(f"Ошибка при вызове LLM: {e}")
            return self._get_default_response()
        return self._parse_llm_response(response)

    def _process_retrieved_docs(self, docs: Any) -> str:
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
        try:
            result = json.loads(response)
            return self._validate_response_structure(result)
        except json.JSONDecodeError:
            self.logger.warning("Не удалось распарсить JSON от LLM")
            return self._get_default_response()

    def _validate_response_structure(self, result: Dict[str, Any]) -> Dict[str, Any]:
        default = self._get_default_response()
        for key, val in default.items():
            if key not in result:
                result[key] = val
        # Проверить meta_info внутренние ключи
        if isinstance(result.get("meta_info"), dict):
            for mk, mv in default["meta_info"].items():
                if mk not in result["meta_info"]:
                    result["meta_info"][mk] = mv
        else:
            result["meta_info"] = default["meta_info"]
        return result

    def _get_default_response(self) -> Dict[str, Any]:
        return {
            "additional_information": [],
            "related_topics": [],
            "follow_up_questions": [],
            "risks_and_limitations": [],
            "meta_info": {"sources": [], "formats": [], "keywords": []}
        }

    async def _postprocess(self, result_data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        return result_data

    def _calculate_confidence(self, result_data: Dict[str, Any], context: AgentContext) -> float:
        if not isinstance(result_data, dict):
            return 0.0
        key_sections = ["additional_information", "related_topics", "follow_up_questions"]
        filled = sum(1 for section in key_sections if isinstance(result_data.get(section), list) and result_data[section])
        if filled >= 3:
            return 0.9
        elif filled > 0:
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