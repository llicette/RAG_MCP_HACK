import json
import asyncio
from typing import Dict, Any, List
from agents.base_agent import BaseAgent, AgentContext, AgentResult, with_retry, with_timeout
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from configs.settings import settings


class AnswerGeneratorAgent(BaseAgent):
    """Агент для генерации ответов на основе извлеченных документов и обогащенного контекста."""
    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("answer_generator", config, langfuse_client)
        from configs.settings import settings
        model_name = self.get_config("model_name", settings.LLM_MODEL_NAME)
        temperature = float(self.get_config("temperature", 0.3))
        base_url = self.get_config("ollama_base_url", None) or str(settings.LLM_BASE_URL)
        self.llm = Ollama(model=model_name, temperature=temperature, base_url=base_url)

        # Параметры извлечения документов
        self.top_k = int(self.get_config("top_k", 5))
        self.max_snippet_chars = int(self.get_config("max_snippet_chars", 500))
        # include_sources можно учитывать в логике цитирования, но структура ответа фиксирована
        self.include_sources = bool(self.get_config("include_sources", True))

        # Шаблон промпта: экранируем {{ и }} для JSON-шаблона
        # PromptTemplate использует {variable}, чтобы вставить question, doc_context, enriched_context.
        # Литеральные { и } в JSON-шаблоне пишем как {{ и }}.
        template = (
            "Ты — эксперт по предоставлению ответов на вопросы на основе корпоративной документации.\n"
            "Вход:\n"
            "Вопрос: {question}\n"
            "Информация из документов (ссылки обозначены в формате [Источник N]):\n"
            "{doc_context}\n"
            "Обогащенный контекст (дополнительная информация, связанные темы, уточняющие вопросы и т.д.):\n"
            "{enriched_context}\n"
            "Задача: на основе представленной информации сформировать полный, точный и понятный ответ на вопрос. "
            "Если возможно, указывай ссылки на документы в тексте ответа в формате [Источник N], где N соответствует порядку из списка документов выше. "
            "Если информация неполна, отметь это и предложи уточнение или дальнейшие шаги.\n"
            "Ответи в формате JSON без лишнего текста, со строго следующей структурой:\n"
            "{{\"answer_text\": \"...\", "
            "\"cited_sources\": [\"Источник 1\", \"Источник 2\"], "
            "\"follow_up_questions\": [\"...\"], "
            "\"used_documents\": [ {{\"source\": \"...\", \"title\": \"...\"}} ]}}\n"
            "Отвечай только валидным JSON."
        )
        self.prompt_template = PromptTemplate(
            input_variables=["question", "doc_context", "enriched_context"],
            template=template
        )

    @with_timeout(120.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        # Извлекаем вопрос
        question = context.original_query or ""
        # Получаем список извлечённых документов: сначала context.documents, затем metadata
        retrieved = None
        if context.documents is not None:
            retrieved = context.documents
        else:
            retrieved = context.metadata.get("retrieved_docs") or context.metadata.get("retrieved_documents")
        if not retrieved or not isinstance(retrieved, list):
            raise ValueError("Отсутствуют извлеченные документы для генерации ответа")

        # Сортируем по score, если поле есть
        try:
            sorted_docs = sorted(retrieved, key=lambda x: x.get("score", 0), reverse=True)
        except Exception:
            sorted_docs = retrieved

        selected_docs = sorted_docs[: self.top_k]

        # Формируем текстовую часть doc_context и список used_documents
        doc_context_items: List[str] = []
        used_documents: List[Dict[str, Any]] = []
        for idx, doc in enumerate(selected_docs, start=1):
            # Определяем source и title
            source = doc.get("source") or doc.get("id") or f"Документ {idx}"
            title = doc.get("title") or source
            # snippet или content
            content = doc.get("snippet") or doc.get("content", "")
            # Обрезаем до max_snippet_chars, убираем переводы строк
            snippet = content.strip().replace("\n", " ")[: self.max_snippet_chars]
            # Форматируем строку: "1. Title: snippet [Источник 1]"
            doc_context_items.append(f"{idx}. {title}: {snippet} [Источник {idx}]")
            used_documents.append({"source": source, "title": title})
        doc_context = "\n".join(doc_context_items)

        # Формируем enriched_context из context.metadata["enriched_context"]
        enriched = context.metadata.get("enriched_context") or {}
        enriched_parts: List[str] = []
        if isinstance(enriched, dict):
            # Дополнительная информация
            ai = enriched.get("additional_information") or []
            if ai and isinstance(ai, list):
                enriched_parts.append("Дополнительная информация: " + "; ".join(str(x) for x in ai))
            # Связанные темы
            rt = enriched.get("related_topics") or []
            if rt and isinstance(rt, list):
                enriched_parts.append("Связанные темы: " + "; ".join(str(x) for x in rt))
            # Уточняющие вопросы
            fu = enriched.get("follow_up_questions") or []
            if fu and isinstance(fu, list):
                enriched_parts.append("Уточняющие вопросы: " + "; ".join(str(x) for x in fu))
            # Риски и ограничения
            rl = enriched.get("risks_and_limitations") or []
            if rl and isinstance(rl, list):
                enriched_parts.append("Риски и ограничения: " + "; ".join(str(x) for x in rl))
            # Мета-информация
            mi = enriched.get("meta_info") or {}
            if isinstance(mi, dict):
                sources = mi.get("sources") or []
                formats = mi.get("formats") or []
                keywords = mi.get("keywords") or []
                meta_items: List[str] = []
                if sources and isinstance(sources, list):
                    meta_items.append("Источники для чтения: " + "; ".join(str(x) for x in sources))
                if formats and isinstance(formats, list):
                    meta_items.append("Рекомендуемые форматы: " + "; ".join(str(x) for x in formats))
                if keywords and isinstance(keywords, list):
                    meta_items.append("Ключевые слова: " + "; ".join(str(x) for x in keywords))
                if meta_items:
                    enriched_parts.append("Мета-информация: " + ". ".join(meta_items))
        enriched_context = "\n".join(enriched_parts) if enriched_parts else "нет"

        # Формируем prompt через PromptTemplate
        prompt = self.prompt_template.format(
            question=question,
            doc_context=doc_context,
            enriched_context=enriched_context
        )

        # Вызов LLM через invoke_llm (retry + timeout)
        response: str
        try:
            response = await self.invoke_llm(prompt)
        except Exception as e:
            self.logger.error(f"Ошибка при вызове LLM в AnswerGeneratorAgent: {e}")
            raise

        # Парсим JSON-ответ
        parsed = self.parse_json_response(response)

        # Если ключ answer_text отсутствует, возвращаем fallback: текст без структурированных полей
        if not isinstance(parsed, dict) or "answer_text" not in parsed:
            self.logger.warning(
                "AnswerGeneratorAgent: не найден ключ 'answer_text' в JSON-ответе от LLM, возвращаем текст без структурированных полей"
            )
            return {
                "answer_text": response.strip(),
                "cited_sources": [],
                "follow_up_questions": [],
                "used_documents": used_documents
            }

        # Убедимся, что ключи есть и имеют правильный тип
        answer_text = parsed.get("answer_text")
        if not isinstance(answer_text, str):
            answer_text = str(answer_text)

        cited_sources = parsed.get("cited_sources") or []
        if not isinstance(cited_sources, list):
            cited_sources = [str(cited_sources)]
        else:
            # Приводим все элементы к строкам
            cited_sources = [str(x) for x in cited_sources]

        follow_up_questions = parsed.get("follow_up_questions") or []
        if not isinstance(follow_up_questions, list):
            follow_up_questions = [str(follow_up_questions)]
        else:
            follow_up_questions = [str(x) for x in follow_up_questions]

        used_docs_from_response = parsed.get("used_documents")
        if used_docs_from_response is None:
            used_documents_final = used_documents
        else:
            # Если LLM вернул used_documents, проверяем, что это список dict с keys source и title
            if isinstance(used_docs_from_response, list):
                cleaned = []
                for item in used_docs_from_response:
                    if isinstance(item, dict):
                        src = item.get("source") or ""
                        title = item.get("title") or ""
                        cleaned.append({"source": str(src), "title": str(title)})
                    else:
                        # Если элемент не dict, пропускаем или добавляем как текст
                        # Здесь просто игнорируем, сохраняя первичный список
                        pass
                # Если cleaned непустой, используем его, иначе fallback на used_documents
                used_documents_final = cleaned if cleaned else used_documents
            else:
                used_documents_final = used_documents

        result: Dict[str, Any] = {
            "answer_text": answer_text,
            "cited_sources": cited_sources,
            "follow_up_questions": follow_up_questions,
            "used_documents": used_documents_final
        }

        return result

    async def _postprocess(self, result_data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        # Здесь можно добавить любые дополнительные проверки/трансформации, сейчас просто возвращаем как есть
        return result_data

    def _calculate_confidence(self, result_data: Any, context: AgentContext) -> float:
        # Базовая логика: если есть непустой answer_text, считаем 0.8, иначе 0.0
        if isinstance(result_data, dict) and result_data.get("answer_text"):
            return 0.8
        return 0.0


def create_answer_generator_agent(config: Dict[str, Any] = None, langfuse_client=None) -> AnswerGeneratorAgent:
    """
    Фабричная функция для создания экземпляра AnswerGeneratorAgent.
    """
    default_config = {
        "model_name": "llama3.1:8b",
        "temperature": 0.3,
        "ollama_base_url": str(settings.LLM_BASE_URL),
        "top_k": 5,
        "max_snippet_chars": 500,
        "include_sources": True
    }
    if config:
        default_config.update(config)
    return AnswerGeneratorAgent(default_config, langfuse_client)
