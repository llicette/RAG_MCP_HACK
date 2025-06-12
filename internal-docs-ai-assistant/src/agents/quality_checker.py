import json
import asyncio
from typing import Dict, Any, List
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from base_agent import BaseAgent, AgentContext, AgentResult, with_retry, with_timeout


class QualityCheckerAgent(BaseAgent):
    """
    Агент для проверки качества сгенерированных ответов.
    Оценивает ответы по нескольким критериям и предлагает улучшения.
    """
    
    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("quality_checker", config, langfuse_client)
        
        # Инициализация LLM с параметрами из конфига
        self.llm = Ollama(
            model=self.get_config("model_name", "llama3.1:8b"),
            temperature=self.get_config("temperature", 0.0),
            base_url=self.get_config("ollama_base_url", "http://localhost:11434")
        )
        
        # Шаблон промпта для проверки качества
        self.check_prompt = PromptTemplate(
            input_variables=["question", "answer", "context_info"],
            template=self._get_check_prompt()
        )

    def _get_check_prompt(self) -> str:
        """
        Возвращает шаблон промпта для оценки качества ответа.
        """
        return ("""
        Ты - эксперт, выполняющий проверку качества ответа на пользовательский запрос.

        Запрос: {question}

        Сгенерированный ответ: {answer}

        Дополнительная информация (источники, документы, контекст), если есть: {context_info}

        Проверь ответ по следующим критериям:
        1. Фактическая точность: есть ли утверждения, требующие проверки или содержащие ошибки.
        2. Полнота: охватывает ли ответ ключевые аспекты запроса, или есть упущения.
        3. Соответствие вопросу: напрямую ли отвечает на запрос, не уходит ли в сторону.
        4. Выявление потенциальных ошибок или противоречий во внутри ответа.
        5. Ясность и читаемость: насколько понятен ответ.
        6. Соответствие стилю пользователя и требованиям компании (если указаны).

        Ответь в JSON формате без лишнего текста:
        {
        "factual_accuracy_score": число_0_1,
        "completeness_score": число_0_1,
        "relevance_score": число_0_1,
        "clarity_score": число_0_1,
        "identified_issues": ["описание проблемы1", "описание проблемы2"],
        "suggestions": ["предложение по улучшению1", "предложение по улучшению2"],
        "overall_quality_score": число_0_1
        }
        """)

    @with_timeout(20.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        """
        Основная логика обработки запроса.
        """
        # Извлечение данных из контекста
        question = context.original_query or ""
        
        if not context.metadata:
            raise ValueError("Отсутствует metadata в контексте")
        
        answer = context.metadata.get("answer")
        context_info = context.metadata.get("context_info", "нет")
        
        if not answer:
            raise ValueError("Отсутствует ответ для проверки в context.metadata['answer']")
        
        # Формирование промпта
        prompt = self.check_prompt.format(
            question=question,
            answer=answer,
            context_info=context_info
        )
        
        # Вызов LLM
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
        except Exception as e:
            self.logger.error(f"Ошибка при вызове LLM: {e}")
            raise
        
        # Парсинг JSON-ответа
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning("Не удалось распарсить JSON от LLM")
            result = self._get_default_result()
        
        return result

    def _get_default_result(self) -> Dict[str, Any]:
        """
        Возвращает дефолтный результат при ошибке парсинга.
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

    async def _postprocess(self, result_data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """
        Дополнительная обработка результата.
        """
        # Здесь можно добавить логику фильтрации, кэширования и т.д.
        return result_data

    def _calculate_confidence(self, result_data: Dict[str, Any], context: AgentContext) -> float:
        """
        Рассчитывает уровень уверенности на основе общего качества ответа.
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
        "ollama_base_url": "http://localhost:11434"
    }
    
    if config:
        default_config.update(config)
        
    return QualityCheckerAgent(default_config, langfuse_client)