import json
from typing import Dict, Any, List, Optional
import asyncio

from agents.base_agent import BaseAgent, AgentContext, with_retry, with_timeout
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from configs.settings import settings


class ContextEnricherAgent(BaseAgent):
    """Агент для обогащения контекста ответа."""

    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("context_enricher", config, langfuse_client)
        # Инициализируем LLM
        model_name = self.get_config("model_name", settings.LLM_MODEL_NAME)
        base_url = self.get_config("ollama_base_url", None) or str(settings.LLM_BASE_URL)
        temperature = float(self.get_config("temperature", 0.1))
        self.llm = Ollama(model=model_name, temperature=temperature, base_url=base_url)

        # PromptTemplate: экранируем {{ }} для JSON-шаблона
        template = (
            "Ты — эксперт по обогащению контекста для ответов на пользовательские запросы.\n"
            "Входные данные:\n"
            "- Запрос: {question}\n"
            "- Текущий ответ: {current_answer}\n"
            "- Извлеченная информация / источники: {retrieved_info}\n"
            "- Дополнительные метаданные: {metadata_info}\n"
            "Задача:\n"
            "1. Добавить релевантную дополнительную информацию или фон, который поможет глубже раскрыть тему запроса.\n"
            "2. Предложить связанные темы или аспекты, которые могут быть интересны пользователю.\n"
            "3. Сформулировать уточняющие или follow-up вопросы, которые могут помочь уточнить запрос или углубить диалог.\n"
            "4. Выделить потенциальные риски или ограничения, связанные с темой (если применимо).\n"
            "5. Предложить мета-информацию: возможные источники для дальнейшего чтения, форматы представления, ключевые термины для поиска.\n"
            "Ответь строго в JSON формате без дополнительного текста, со следующей структурой:\n"
            "{{"
            "\"additional_information\": [\"...\"], "
            "\"related_topics\": [\"...\"], "
            "\"follow_up_questions\": [\"...\"], "
            "\"risks_and_limitations\": [\"...\"], "
            "\"meta_info\": {{\"sources\": [\"...\"], \"formats\": [\"...\"], \"keywords\": [\"...\"]}}"
            "}}"
        )
        self.enrich_prompt = PromptTemplate(
            input_variables=["question", "current_answer", "retrieved_info", "metadata_info"],
            template=template
        )

    @with_timeout(60.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        # Извлекаем исходные данные из context
        question = context.original_query or ""
        metadata = context.metadata or {}

        # Текущий ответ: может быть ключ "answer_text" или "answer" в metadata
        current_answer = metadata.get("answer_text")
        if not isinstance(current_answer, str):
            current_answer = metadata.get("answer") or "нет"
        if not isinstance(current_answer, str):
            current_answer = "нет"

        # Извлечённая информация: берём context.documents или metadata["retrieved_docs"]
        retrieved = None
        if context.documents is not None:
            retrieved = context.documents
        else:
            retrieved = metadata.get("retrieved_docs") or metadata.get("retrieved_documents")
        retrieved_info = self._process_retrieved_docs(retrieved)

        # Дополнительные метаданные: любые поля из metadata, кроме ключей, уже использованных
        # Представляем metadata_info как строку JSON или str()
        # Чтобы не передавать весь metadata, можем передать ключ "additional_context", если есть
        additional_context = metadata.get("additional_context")
        if additional_context is None:
            # Уберём большие поля: соберём ключи кроме retrieved_docs и answer*
            small_meta = {}
            for k, v in metadata.items():
                if k in ("retrieved_docs", "retrieved_documents", "answer_text", "answer"):
                    continue
                # Если v сериализуемо коротко
                small_meta[k] = v
            try:
                metadata_info = json.dumps(small_meta, ensure_ascii=False)
            except Exception:
                metadata_info = str(small_meta)
        else:
            try:
                metadata_info = json.dumps(additional_context, ensure_ascii=False)
            except Exception:
                metadata_info = str(additional_context)

        # Формируем prompt
        prompt = self.enrich_prompt.format(
            question=question,
            current_answer=current_answer,
            retrieved_info=retrieved_info,
            metadata_info=metadata_info
        )

        # Вызов LLM через invoke_llm
        try:
            response = await self.invoke_llm(prompt)
        except Exception as e:
            self.logger.error(f"ContextEnricherAgent: ошибка при вызове LLM: {e}")
            return self._get_default_response()

        # Парсим JSON-ответ
        parsed = self.parse_json_response(response)
        if not isinstance(parsed, dict):
            self.logger.warning("ContextEnricherAgent: LLM ответ не JSON-объект")
            return self._get_default_response()
        return self._validate_response_structure(parsed)

    def _process_retrieved_docs(self, docs: Any) -> str:
        if not docs or not isinstance(docs, list):
            return "нет"
        snippets: List[str] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            title = doc.get("title") or doc.get("source") or doc.get("id") or ""
            content = doc.get("snippet") or doc.get("content", "")
            # Обрезаем до первых ~200 символов (или настраиваемый параметр)
            snippet = content.strip().replace("\n", " ")
            snippet = snippet[:200]
            if title:
                snippets.append(f"{title}: {snippet}")
            else:
                snippets.append(snippet)
        if not snippets:
            return "нет"
        # Соединяем точкой с запятой
        return "; ".join(snippets)

    def _get_default_response(self) -> Dict[str, Any]:
        return {
            "additional_information": [],
            "related_topics": [],
            "follow_up_questions": [],
            "risks_and_limitations": [],
            "meta_info": {"sources": [], "formats": [], "keywords": []}
        }

    def _validate_response_structure(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Убедимся, что в словаре result есть все нужные ключи, и они корректного типа.
        Иначе добавляем пустые/дефолтные значения.
        """
        default = self._get_default_response()
        # Проверяем верхний уровень
        out: Dict[str, Any] = {}
        for key in default:
            val = result.get(key)
            if key != "meta_info":
                if isinstance(val, list):
                    # Приводим элементы к строкам
                    out[key] = [str(x) for x in val]
                else:
                    out[key] = []  # если отсутствует или неверный тип
            else:
                # meta_info: ожидаем dict с keys "sources","formats","keywords" и списками
                mi = result.get("meta_info")
                if isinstance(mi, dict):
                    sources = mi.get("sources")
                    formats = mi.get("formats")
                    keywords = mi.get("keywords")
                    # Приведение и дефолты
                    if isinstance(sources, list):
                        s_list = [str(x) for x in sources]
                    else:
                        s_list = []
                    if isinstance(formats, list):
                        f_list = [str(x) for x in formats]
                    else:
                        f_list = []
                    if isinstance(keywords, list):
                        k_list = [str(x) for x in keywords]
                    else:
                        k_list = []
                    out["meta_info"] = {"sources": s_list, "formats": f_list, "keywords": k_list}
                else:
                    out["meta_info"] = default["meta_info"]
        return out

    async def _postprocess(self, result_data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        # Здесь можно добавить дополнительное логирование или сохранение в metadata
        # Например, записать обогащенный контекст в metadata для последующих агентов:
        context.metadata["enriched_context"] = result_data
        return result_data

    def _calculate_confidence(self, result_data: Dict[str, Any], context: AgentContext) -> float:
        if not isinstance(result_data, dict):
            return 0.0
        # Оцениваем, сколько разделов заполнено непустыми списками
        filled = 0
        for key in ("additional_information", "related_topics", "follow_up_questions"):
            val = result_data.get(key)
            if isinstance(val, list) and val:
                filled += 1
        # Чем больше заполнено, тем выше уверенность
        if filled >= 3:
            return 0.9
        elif filled == 2:
            return 0.7
        elif filled == 1:
            return 0.5
        else:
            return 0.2


def create_context_enricher_agent(config: Dict[str, Any] = None, langfuse_client=None) -> ContextEnricherAgent:
    """
    Фабричная функция создания ContextEnricherAgent.
    """
    default_config = {
        "model_name": settings.LLM_MODEL_NAME,
        "ollama_base_url": str(settings.LLM_BASE_URL),
        "temperature": 0.1
    }
    if config:
        default_config.update(config)
    return ContextEnricherAgent(default_config, langfuse_client)
