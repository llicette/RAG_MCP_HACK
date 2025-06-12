"""
Agent для переписывания и улучшения поисковых запросов
"""
import re
import json
import asyncio
from typing import Dict, Any, List, Optional
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

from agents.base_agent import BaseAgent, AgentContext, with_retry, with_timeout

class QueryRewriterAgent(BaseAgent):
    """Агент для переписывания запросов для лучшего поиска"""
    
    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("query_rewriter", config, langfuse_client)
        
        self.llm = Ollama(
            model=self.get_config("model_name", "llama3.1:8b"),
            temperature=self.get_config("temperature", 0.2),
            base_url=self.get_config("ollama_base_url", "http://localhost:11434")
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["original_query", "context", "critic_analysis"],
            template=self._get_prompt_template()
        )
        
        # Словарь сокращений и их расшифровок
        self.abbreviations = {
            "тз": "техническое задание",
            "кр": "код ревью", 
            "пр": "пулл реквест",
            "дб": "база данных",
            "апи": "API",
            "фе": "фронтенд",
            "бе": "бэкенд",
            "ци": "CI/CD",
            "дев": "разработка",
            "прод": "продакшн",
            "стейдж": "staging",
            "дока": "документация",
            "репо": "репозиторий",
            "коммит": "commit",
            "мердж": "merge",
            "бранч": "ветка",
            "релиз": "release"
        }
        
        # Синонимы для расширения поиска
        self.synonyms = {
            "ошибка": ["баг", "проблема", "неисправность", "дефект"],
            "инструкция": ["руководство", "гайд", "мануал", "как сделать"],
            "настройка": ["конфигурация", "параметры", "установки"],
            "развертывание": ["деплой", "установка", "запуск"],
            "тестирование": ["тесты", "проверка", "валидация"],
            "безопасность": ["секьюрити", "защита", "авторизация"],
            "производительность": ["перформанс", "скорость", "оптимизация"]
        }
    
    def _get_prompt_template(self) -> str:
        """Получение шаблона промпта для переписывания запросов"""
        return """Ты - эксперт по оптимизации поисковых запросов для системы внутренней документации.

Оригинальный запрос пользователя: "{original_query}"

Анализ критика вопросов: {critic_analysis}

Контекст: {context}

Твоя задача - переписать запрос для лучшего поиска по документации, учитывая:
1. Расширение сокращений
2. Добавление синонимов
3. Декомпозицию сложных запросов
4. Устранение неоднозначностей
5. Оптимизацию для семантического поиска

Создай несколько вариантов запросов в JSON формате:

{{
  "rewritten_query": "основной улучшенный запрос",
  "alternative_queries": [
    "альтернативный запрос 1",
    "альтернативный запрос 2",
    "альтернативный запрос 3"
  ],
  "search_keywords": [
    "ключевое слово 1",
    "ключевое слово 2",
    "ключевое слово 3"
  ],
  "expanded_terms": {{
    "оригинальный_термин": "расширенный_термин"
  }},
  "query_type": "factual|procedural|troubleshooting|conceptual",
  "search_strategy": "semantic|keyword|hybrid",
  "filters": {{
    "document_types": ["тип1", "тип2"],
    "topics": ["тема1", "тема2"],
    "departments": ["отдел1", "отдел2"]
  }},
  "decomposed_queries": [
    "подзапрос 1",
    "подзапрос 2"
  ],
  "confidence": 0.0-1.0,
  "improvements_made": [
    "описание улучшения 1",
    "описание улучшения 2"
  ]
}}

Отвечай только валидным JSON."""
    
    @with_timeout(20.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        """Основная логика переписывания запроса"""
        original_query = context.processed_query or context.original_query
        
        # Предварительная обработка
        preprocessed = self._preprocess_query(original_query)
        
        # Анализ критика (если есть)
        critic_analysis = self._extract_critic_analysis(context)
        
        # Переписывание с помощью LLM
        llm_result = await self._llm_rewrite(preprocessed, context, critic_analysis)
        
        # Постобработка и валидация
        final_result = self._postprocess_result(llm_result, preprocessed, context)
        
        return final_result

    def _preprocess_query(self, query: str) -> str:
        """Предварительная обработка запроса: расширение сокращений"""
        # Расширение сокращений
        for abbr, full in self.abbreviations.items():
            query = re.sub(r'\b' + re.escape(abbr) + r'\b', full, query, flags=re.IGNORECASE)
        
        # Добавление пробелов вокруг специальных символов
        query = re.sub(r'([/-])', r' \1 ', query)
        
        return query.strip()

    def _extract_critic_analysis(self, context: AgentContext) -> str:
        """Извлечение анализа критика из контекста"""
        if hasattr(context, 'critic_analysis'):
            return context.critic_analysis
        return "Нет критического анализа предыдущих запросов."

    async def _llm_rewrite(self, preprocessed: str, context: AgentContext, 
                          critic_analysis: str) -> Dict[str, Any]:
        """Переписывание запроса с помощью LLM"""
        try:
            prompt = self.prompt_template.format(
                original_query=context.original_query,
                context=context.context or "",
                critic_analysis=critic_analysis
            )
            
            response = await self.llm.apredict(prompt)
            
            # Попытка преобразовать ответ в JSON
            try:
                result = json.loads(response.strip())
            except json.JSONDecodeError:
                # Логирование ошибки парсинга
                self.logger.error(f"LLM вернул невалидный JSON: {response}")
                result = {
                    "rewritten_query": preprocessed,
                    "improvements_made": ["Не удалось получить ответ от LLM, использован предварительный запрос"],
                    "confidence": 0.6
                }
            
            return result
            
        except Exception as e:
            # Обработка ошибок LLM
            self.logger.error(f"Ошибка при вызове LLM: {e}")
            return {
                "rewritten_query": preprocessed,
                "improvements_made": ["Ошибка LLM, использован предварительный запрос"],
                "confidence": 0.5
            }

    def _postprocess_result(self, result: Dict[str, Any], 
                           preprocessed: str, context: AgentContext) -> Dict[str, Any]:
        """Постобработка и валидация результатов"""
        # Гарантия наличия всех необходимных полей
        required_fields = {
            "rewritten_query": preprocessed,
            "alternative_queries": [],
            "search_keywords": [],
            "expanded_terms": {},
            "query_type": "factual",
            "search_strategy": "hybrid",
            "filters": {"document_types": [], "topics": [], "departments": []},
            "decomposed_queries": [],
            "confidence": 0.7,
            "improvements_made": ["Базовое переписывание запроса"]
        }
        
        # Заполнение отсутствующих полей
        for field, default in required_fields.items():
            if field not in result:
                result[field] = default
        
        # Добавление расширенных терминов
        result["expanded_terms"].update(self._extract_expanded_terms(context.original_query, preprocessed))
        
        # Добавление синонимов в альтернативные запросы
        synonyms_queries = self._generate_alternative_queries(context.original_query)
        result["alternative_queries"].extend(synonyms_queries)
        
        # Удаление дубликатов в альтернативных запросах
        result["alternative_queries"] = list(dict.fromkeys(result["alternative_queries"]))
        
        # Определение типа запроса, если не указан
        if result["query_type"] == "factual":
            result["query_type"] = self._determine_query_type(context.original_query)
        
        # Добавление ключевых слов из оригинального запроса
        if not result["search_keywords"]:
            result["search_keywords"] = self._extract_keywords(context.original_query)
        
        return result

    def _extract_expanded_terms(self, original: str, preprocessed: str) -> Dict[str, str]:
        """Извлечение расширенных терминов из запроса"""
        expanded = {}
        for abbr, full in self.abbreviations.items():
            if re.search(r'\b' + re.escape(abbr) + r'\b', original, re.IGNORECASE):
                if full.lower() in preprocessed.lower():
                    expanded[abbr] = full
        return expanded

    def _generate_alternative_queries(self, query: str) -> List[str]:
        """Генерация альтернативных запросов на основе синонимов"""
        alternatives = []
        query_lower = query.lower()
        
        for keyword, syns in self.synonyms.items():
            if keyword in query_lower:
                for syn in syns:
                    # Замена с сохранением регистра
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                    alt = pattern.sub(syn, query)
                    alternatives.append(alt)
        
        return list(set(alternatives))

    def _determine_query_type(self, query: str) -> str:
        """Определение типа запроса"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["как", "инструкция", "руководство", "сделать"]):
            return "procedural"
        elif any(word in query_lower for word in ["ошибка", "не работает", "проблема", "баг"]):
            return "troubleshooting"
        elif any(word in query_lower for word in ["что такое", "описание", "что это", "концепция"]):
            return "conceptual"
        else:
            return "factual"

    def _extract_keywords(self, query: str) -> List[str]:
        """Извлечение ключевых слов из запроса"""
        # Простая реализация - разбиение на слова и удаление стоп-слов
        stop_words = {"как", "где", "когда", "почему", "что", "это", "на", "в", "и", "или", "не"}
        words = re.findall(r'\b\w+\b', query.lower())
        return [word for word in words if word not in stop_words]
    
# Фабрика для создания агента
def create_question_critic_agent(config: Dict[str, Any] = None, 
                                langfuse_client=None) -> QueryRewriterAgent:
    """Фабричная функция для создания агента переписывания запросов"""
    default_config = {
        "model_name": "llama3.1:8b",
        "temperature": 0.3,
        "ollama_base_url": "http://localhost:11434",
        "max_analysis_time": 15.0,
        "enable_pattern_analysis": True,
        "enable_llm_analysis": True
    }
    
    if config:
        default_config.update(config)
    
    return QueryRewriterAgent(default_config, langfuse_client)