import json
from typing import Dict, Any
import asyncio

from agents.base_agent import BaseAgent, AgentContext, AgentResult, with_retry, with_timeout
# Используйте новый импорт Ollama, если LangChain обновлён
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from configs.settings import settings


class QualityCheckerAgent(BaseAgent):
    """
    Агент для проверки качества сгенерированных ответов.
    Оценивает ответы по нескольким критериям и предлагает улучшения.
    """
    
    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("quality_checker", config, langfuse_client)
        
        # Инициализация LLM через Ollama с конфигом
        model_name = self.get_config("model_name", settings.LLM_MODEL_NAME)
        base_url = self.get_config("ollama_base_url", None) or str(settings.LLM_BASE_URL)
        temperature = float(self.get_config("temperature", 0.0))
        self.llm = Ollama(model=model_name, temperature=temperature, base_url=base_url)
        
        # PromptTemplate: экранируем JSON-структуру двойными {{ }}
        template = (
            "Ты — эксперт, выполняющий проверку качества ответа на пользовательский запрос.\n"
            "Запрос: {question}\n"
            "Сгенерированный ответ: {answer}\n"
            "Дополнительная информация (извлечённые документы, обогащённый контекст), если есть: {context_info}\n"
            "Проверь ответ по следующим критериям:\n"
            "1. Фактическая точность: есть ли утверждения, требующие проверки или содержащие ошибки.\n"
            "2. Полнота: охватывает ли ответ ключевые аспекты запроса, или есть упущения.\n"
            "3. Соответствие вопросу: напрямую ли отвечает на запрос, не уходит ли в сторону.\n"
            "4. Выявление потенциальных ошибок или противоречий внутри ответа.\n"
            "5. Ясность и читаемость: насколько понятен ответ.\n"
            "6. Соответствие стилю пользователя и требованиям компании (если указаны).\n"
            "Ответь строго в JSON формате без лишнего текста, со структурой:\n"
            "{{"
            "\"factual_accuracy_score\": число_0_1, "
            "\"completeness_score\": число_0_1, "
            "\"relevance_score\": число_0_1, "
            "\"clarity_score\": число_0_1, "
            "\"identified_issues\": [\"описание проблемы1\", \"описание проблемы2\"], "
            "\"suggestions\": [\"предложение по улучшению1\", \"предложение по улучшению2\"], "
            "\"overall_quality_score\": число_0_1"
            "}}"
        )
        self.check_prompt = PromptTemplate(
            input_variables=["question", "answer", "context_info"],
            template=template
        )

    @with_timeout(60.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        """
        Основная логика: берём из context оригинальный запрос, ответ и контекст, 
        формируем prompt, вызываем LLM и парсим JSON.
        """
        question = context.original_query or ""
        
        # Извлекаем ответ для проверки. Возможные поля: answer_text, final_answer, answer
        answer = None
        # Обычно AnswerGenerator кладёт результат в result_data["answer_text"], а orchestrator может сохранить в metadata["final_answer"]
        if context.metadata:
            ans = context.metadata.get("answer_text")
            if isinstance(ans, str) and ans.strip():
                answer = ans.strip()
            else:
                ans2 = context.metadata.get("final_answer")
                if isinstance(ans2, str) and ans2.strip():
                    answer = ans2.strip()
                else:
                    ans3 = context.metadata.get("answer")
                    if isinstance(ans3, str) and ans3.strip():
                        answer = ans3.strip()
        if not answer:
            raise ValueError("Отсутствует сгенерированный ответ для проверки (metadata['answer_text'] или ['final_answer'] или ['answer'])")
        
        # Формируем context_info: комбинируем извлечённые документы и обогащённый контекст, если имеются
        context_info_parts = []
        # Извлечённые документы: обычно в context.documents или context.metadata["retrieved_docs"]
        retrieved = None
        if context.documents is not None:
            retrieved = context.documents
        else:
            retrieved = context.metadata.get("retrieved_docs") or context.metadata.get("retrieved_documents")
        if isinstance(retrieved, list) and retrieved:
            # Формируем краткое описание: title + snippet (первые ~200 символов) для каждого
            snippets = []
            for idx, doc in enumerate(retrieved, start=1):
                if not isinstance(doc, dict):
                    continue
                title = doc.get("title") or doc.get("source") or doc.get("id") or f"Документ{idx}"
                content = doc.get("snippet") or doc.get("content", "")
                snippet = content.strip().replace("\n", " ")
                snippet = snippet[:200]
                snippets.append(f"{idx}. {title}: {snippet}")
            if snippets:
                context_info_parts.append("Извлечённые документы:\n" + "\n".join(snippets))
        # Обогащённый контекст: обычно в context.metadata["enriched_context"]
        enriched = context.metadata.get("enriched_context")
        if isinstance(enriched, dict) and enriched:
            try:
                enriched_str = json.dumps(enriched, ensure_ascii=False)
            except Exception:
                enriched_str = str(enriched)
            context_info_parts.append("Обогащённый контекст:\n" + enriched_str)
        # Результаты предыдущих проверок или другие поля context.metadata["additional_context_for_quality"], если есть
        additional = context.metadata.get("additional_context_for_quality")
        if additional:
            try:
                additional_str = json.dumps(additional, ensure_ascii=False)
            except Exception:
                additional_str = str(additional)
            context_info_parts.append("Дополнительная информация:\n" + additional_str)
        
        context_info = "\n---\n".join(context_info_parts) if context_info_parts else "нет"
        
        # Формируем prompt через PromptTemplate
        prompt = self.check_prompt.format(
            question=question,
            answer=answer,
            context_info=context_info
        )
        
        # Вызов LLM через invoke_llm (retry + таймаут)
        try:
            response = await self.invoke_llm(prompt)
        except Exception as e:
            self.logger.error(f"QualityCheckerAgent: ошибка при вызове LLM: {e}")
            # Вернём дефолтный результат
            return self._get_default_result()
        
        # Парсинг JSON-ответа
        parsed = self.parse_json_response(response)
        if not isinstance(parsed, dict):
            self.logger.warning("QualityCheckerAgent: LLM ответ не JSON-объект, возвращаем дефолт")
            result = self._get_default_result()
        else:
            # Проверяем наличие обязательных ключей; если отсутствуют, подставляем дефолт для каждого
            result = self._validate_response(parsed)
        return result

    def _get_default_result(self) -> Dict[str, Any]:
        """
        Возвращает дефолтный результат при ошибке парсинга или LLM-вызова.
        """
        return {
            "factual_accuracy_score": 0.0,
            "completeness_score": 0.0,
            "relevance_score": 0.0,
            "clarity_score": 0.0,
            "identified_issues": ["Не удалось распознать структуру ответа"],
            "suggestions": ["Повторите проверку с корректным JSON-промптом"],
            "overall_quality_score": 0.0
        }

    def _validate_response(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Убедимся, что в parsed есть все требуемые ключи и они корректного типа.
        Если их нет или тип неверный, подставляем дефолт из _get_default_result.
        """
        default = self._get_default_result()
        out: Dict[str, Any] = {}
        # Проверяем числовые поля
        for key in ("factual_accuracy_score", "completeness_score", "relevance_score", "clarity_score", "overall_quality_score"):
            val = parsed.get(key)
            if isinstance(val, (int, float)):
                # Приводим к диапазону [0.0, 1.0]
                try:
                    fv = float(val)
                    out[key] = max(0.0, min(1.0, fv))
                except:
                    out[key] = default[key]
            else:
                out[key] = default[key]
        # identified_issues: должен быть список строк
        issues = parsed.get("identified_issues")
        if isinstance(issues, list):
            out["identified_issues"] = [str(x) for x in issues]
        else:
            out["identified_issues"] = default["identified_issues"]
        # suggestions: список строк
        sugg = parsed.get("suggestions")
        if isinstance(sugg, list):
            out["suggestions"] = [str(x) for x in sugg]
        else:
            out["suggestions"] = default["suggestions"]
        return out

    async def _postprocess(self, result_data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """
        Сохраняем результат проверки в context.metadata, чтобы другие агенты или orchestrator могли его использовать.
        Например: context.metadata['quality_check'] = result_data
        """
        context.metadata["quality_check"] = result_data
        # На случай, если orchestrator ожидает флаг needs_query_improvement или needs_context_enrichment:
        # можем выставить подсказки:
        overall = result_data.get("overall_quality_score", 0.0)
        # Пример: если общая оценка ниже 0.7, предлагаем retry или context enrichment:
        if isinstance(overall, (int, float)) and overall < 0.7:
            # Например, вставляем подсказки:
            # Если фактическая точность низкая: needs_context_enrichment = True
            if result_data.get("factual_accuracy_score", 0.0) < 0.7:
                context.metadata.setdefault("quality_hints", {})["needs_context_enrichment"] = True
            # Если релевантность низкая: needs_query_improvement = True
            if result_data.get("relevance_score", 0.0) < 0.7:
                context.metadata.setdefault("quality_hints", {})["needs_query_improvement"] = True
        return result_data

    def _calculate_confidence(self, result_data: Any, context: AgentContext) -> float:
        """
        Рассчитывает уровень уверенности агента на основе overall_quality_score.
        """
        if isinstance(result_data, dict):
            score = result_data.get("overall_quality_score")
            if isinstance(score, (int, float)):
                return max(0.0, min(1.0, float(score)))
        return 0.0


def create_quality_checker_agent(config: Dict[str, Any] = None, langfuse_client=None) -> QualityCheckerAgent:
    """
    Фабричная функция для создания экземпляра QualityCheckerAgent.
    """
    default_config = {
        "model_name": "llama3.1:8b",
        "temperature": 0.0,
        "ollama_base_url": str(settings.LLM_BASE_URL)
    }
    if config:
        default_config.update(config)
    return QualityCheckerAgent(default_config, langfuse_client)
