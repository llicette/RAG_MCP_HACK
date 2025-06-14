import re
import json
import asyncio
from typing import Dict, Any, List
from agents.base_agent import BaseAgent, AgentContext, with_retry, with_timeout
from langchain_community.llms import Ollama  # Или from langchain.llms import Ollama, если требуется
from langchain.prompts import PromptTemplate
from configs.settings import settings  # путь к settings, скорректируйте при необходимости


class QueryRewriterAgent(BaseAgent):
    """Агент для переписывания запросов для лучшего поиска"""

    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("query_rewriter", config, langfuse_client)

        # Настройка LLM через Ollama
        model_name = self.get_config("model_name", settings.LLM_MODEL_NAME)
        base_url = self.get_config("ollama_base_url", None) or str(settings.LLM_BASE_URL)
        temperature = float(self.get_config("temperature", 0.1))
        self.llm = Ollama(model=model_name, temperature=temperature, base_url=base_url)

        # PromptTemplate: экранируем фигурные скобки {{ }} для JSON-образца
        prompt_template_str = self._get_prompt_template()
        self.prompt_template = PromptTemplate(
            input_variables=["original_query", "context_str", "critic_analysis"],
            template=prompt_template_str
        )

        # Словарь сокращений -> полные формы
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

        # Синонимы для альтернативных запросов
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
        """Шаблон PromptTemplate для переписывания запросов."""
        # Экранируем JSON-образец двойными {{ }}
        return """Ты — эксперт по оптимизации поисковых запросов для системы внутренней документации.

Оригинальный запрос пользователя: "{original_query}"

Критический анализ запроса (если имеется): {critic_analysis}

Контекст (например, тема или предыдущий ответ): {context_str}

Твоя задача — переписать запрос для лучшего поиска по документации, учитывая:
1. Расширение сокращений
2. Добавление синонимов
3. Декомпозицию сложных запросов
4. Устранение неоднозначностей
5. Оптимизацию для семантического поиска

Создай несколько вариантов запросов в JSON формате:

{{
  "rewritten_query": "основной улучшенный запрос",
  "alternative_queries": ["альтернативный запрос 1", "альтернативный запрос 2", "альтернативный запрос 3"],
  "search_keywords": ["ключевое слово 1", "ключевое слово 2", "ключевое слово 3"],
  "expanded_terms": {{"оригинальный_термин": "расширенный_термин"}},
  "query_type": "factual|procedural|troubleshooting|conceptual",
  "search_strategy": "semantic|keyword|hybrid",
  "filters": {{"document_types": ["тип1", "тип2"], "topics": ["тема1", "тема2"], "departments": ["отдел1", "отдел2"]}},
  "decomposed_queries": ["подзапрос 1", "подзапрос 2"],
  "confidence": 0.0,
  "improvements_made": ["описание улучшения 1", "описание улучшения 2"]
}}

Отвечай только валидным JSON без дополнительного текста."""
    
    @with_timeout(60.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        """
        Основная логика переписывания запроса.
        """
        # Оригинальный запрос: processed_query или original_query
        original_query = context.processed_query or context.original_query or ""
        # Предварительная обработка: расширение сокращений и прочее
        preprocessed_query = self._preprocess_query(original_query)

        # Извлечь критический анализ из context.metadata, если был записан QuestionCriticAgent
        critic_analysis = self._extract_critic_analysis(context)

        # Формируем контекст: например, тема из context.topic и/или краткое содержание контекста
        context_str = self._build_context_str(context)

        # Вызов LLM для переписывания
        llm_result = await self._llm_rewrite(preprocessed_query, context_str, original_query, critic_analysis)

        # Постобработка: заполнение обязательных полей, добавление расширенных терминов, альтернативных запросов и т.д.
        final_result = self._postprocess_result(llm_result, original_query, preprocessed_query, context)

        return final_result

    def _preprocess_query(self, query: str) -> str:
        """
        Предварительная обработка: расширение сокращений, нормализация пробелов.
        """
        q = query
        # Расширение сокращений
        for abbr, full in self.abbreviations.items():
            # Замена целиком слова, игнорируя регистр
            q = re.sub(r'\b' + re.escape(abbr) + r'\b', full, q, flags=re.IGNORECASE)
        # Добавление пробелов вокруг специальных символов для лучшей токенизации
        q = re.sub(r'([/-])', r' \1 ', q)
        return q.strip()

    def _extract_critic_analysis(self, context: AgentContext) -> str:
        """
        Попытка извлечь результат QuestionCriticAgent или другие подсказки из metadata.
        Например, context.metadata["critic_analysis"] или context.metadata["quality_hints"].
        """
        # Пример: если в metadata сохранили под ключом "critic_analysis"
        critic = context.metadata.get("critic_analysis")
        if isinstance(critic, str) and critic.strip():
            return critic.strip()
        # Возможно, QuestionCriticAgent сохранил информацию в context.metadata["agent_results"]["question_critic"]
        ar = context.metadata.get("agent_results")
        if isinstance(ar, dict):
            qc = ar.get("question_critic")
            if isinstance(qc, dict):
                # Попытаемся взять текстовое поле из анализа
                text = qc.get("analysis") or qc.get("reasoning") or qc.get("feedback")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        # Нет анализа
        return "Нет критического анализа предыдущего запроса."

    def _build_context_str(self, context: AgentContext) -> str:
        """
        Формирует строку контекста: можно взять topic, краткий summary предыдущих ответов или metadata.
        """
        parts: List[str] = []
        # Тема из классификатора
        if context.topic:
            parts.append(f"Тема: {context.topic}")
        # Если есть обогащённый контекст из ContextEnricherAgent
        enriched = context.metadata.get("enriched_context")
        if isinstance(enriched, dict) and enriched:
            # Берём ключи, формируем короткую строку
            try:
                enriched_str = json.dumps(enriched, ensure_ascii=False)
            except:
                enriched_str = str(enriched)
            parts.append(f"Обогащённый контекст: {enriched_str}")
        # Если есть последний ответ, можно добавить кратко
        last_answer = context.metadata.get("answer_text") or context.metadata.get("final_answer")
        if isinstance(last_answer, str) and last_answer.strip():
            snippet = last_answer.strip().replace("\n", " ")
            snippet = snippet[:200]
            parts.append(f"Последний ответ: {snippet}")
        # Объединяем
        if parts:
            return " | ".join(parts)
        else:
            return "Нет дополнительного контекста."

    async def _llm_rewrite(self, preprocessed_query: str, context_str: str, original_query: str, critic_analysis: str) -> Dict[str, Any]:
        """
        Вызов LLM для переписывания запроса.
        """
        try:
            prompt = self.prompt_template.format(
                original_query=original_query,
                context_str=context_str,
                critic_analysis=critic_analysis
            )
            # Вызываем LLM через invoke_llm (retry + timeout)
            response = await self.invoke_llm(prompt)
        except Exception as e:
            self.logger.error(f"QueryRewriterAgent: ошибка при вызове LLM: {e}")
            # Возвращаем минимальный результат
            return {
                "rewritten_query": preprocessed_query,
                "alternative_queries": [],
                "search_keywords": [],
                "expanded_terms": {},
                "query_type": "factual",
                "search_strategy": "hybrid",
                "filters": {"document_types": [], "topics": [], "departments": []},
                "decomposed_queries": [],
                "confidence": 0.5,
                "improvements_made": ["Ошибка LLM, использован предварительный запрос"]
            }

        # Парсим JSON-ответ
        parsed = self.parse_json_response(response.strip())
        if not isinstance(parsed, dict):
            # Логируем проблему
            self.logger.warning(f"QueryRewriterAgent: LLM вернул не JSON-объект: {response}")
            return {
                "rewritten_query": preprocessed_query,
                "alternative_queries": [],
                "search_keywords": [],
                "expanded_terms": {},
                "query_type": "factual",
                "search_strategy": "hybrid",
                "filters": {"document_types": [], "topics": [], "departments": []},
                "decomposed_queries": [],
                "confidence": 0.6,
                "improvements_made": ["Не удалось распознать JSON от LLM, использован предварительный запрос"]
            }
        return parsed

    def _postprocess_result(self, result: Dict[str, Any], original_query: str, preprocessed_query: str, context: AgentContext) -> Dict[str, Any]:
        """
        Заполнение обязательных полей, удаление дубликатов, добавление расширенных терминов и альтернативных запросов.
        """
        # Определяем набор обязательных полей с дефолтными значениями
        required_fields = {
            "rewritten_query": preprocessed_query,
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
        # Заполняем отсутствующие поля дефолтами
        for field, default in required_fields.items():
            if field not in result or result[field] is None:
                result[field] = default
        # Приведение типов: 
        # rewritten_query
        if not isinstance(result["rewritten_query"], str) or not result["rewritten_query"].strip():
            result["rewritten_query"] = preprocessed_query
        # alternative_queries: список строк
        if isinstance(result.get("alternative_queries"), list):
            alt = []
            for x in result["alternative_queries"]:
                if isinstance(x, str) and x.strip():
                    alt.append(x.strip())
            result["alternative_queries"] = alt
        else:
            result["alternative_queries"] = []
        # search_keywords: список строк
        if isinstance(result.get("search_keywords"), list):
            kws = []
            for x in result["search_keywords"]:
                if isinstance(x, str) and x.strip():
                    kws.append(x.strip())
            result["search_keywords"] = kws
        else:
            result["search_keywords"] = []
        # expanded_terms: dict str->str
        if isinstance(result.get("expanded_terms"), dict):
            et = {}
            for k, v in result["expanded_terms"].items():
                if isinstance(k, str) and isinstance(v, str):
                    et[k] = v
            result["expanded_terms"] = et
        else:
            result["expanded_terms"] = {}
        # query_type
        qt = result.get("query_type")
        if not isinstance(qt, str) or qt not in {"factual", "procedural", "troubleshooting", "conceptual"}:
            result["query_type"] = self._determine_query_type(original_query)
        # search_strategy
        ss = result.get("search_strategy")
        if not isinstance(ss, str) or ss not in {"semantic", "keyword", "hybrid"}:
            result["search_strategy"] = "hybrid"
        # filters
        fl = result.get("filters")
        if not isinstance(fl, dict):
            result["filters"] = {"document_types": [], "topics": [], "departments": []}
        else:
            # Убедимся, что внутри списки
            dt = fl.get("document_types")
            tp = fl.get("topics")
            dp = fl.get("departments")
            result["filters"] = {
                "document_types": dt if isinstance(dt, list) else [],
                "topics": tp if isinstance(tp, list) else [],
                "departments": dp if isinstance(dp, list) else []
            }
        # decomposed_queries: список строк
        dq = result.get("decomposed_queries")
        if isinstance(dq, list):
            dq_list = []
            for x in dq:
                if isinstance(x, str) and x.strip():
                    dq_list.append(x.strip())
            result["decomposed_queries"] = dq_list
        else:
            result["decomposed_queries"] = []
        # confidence
        conf = result.get("confidence")
        if isinstance(conf, (int, float)):
            try:
                conf_f = float(conf)
                result["confidence"] = max(0.0, min(1.0, conf_f))
            except:
                result["confidence"] = 0.7
        else:
            result["confidence"] = 0.7
        # improvements_made: список строк
        im = result.get("improvements_made")
        if isinstance(im, list):
            imp = []
            for x in im:
                if isinstance(x, str) and x.strip():
                    imp.append(x.strip())
            result["improvements_made"] = imp if imp else ["Базовое переписывание запроса"]
        else:
            result["improvements_made"] = ["Базовое переписывание запроса"]

        # Добавляем расширенные термины на основании сокращений
        expanded = self._extract_expanded_terms(original_query, preprocessed_query)
        # Объединяем с тем, что пришло
        result["expanded_terms"].update(expanded)

        # Добавляем альтернативные запросы на основании синонимов
        alt_syns = self._generate_alternative_queries(original_query)
        # Добавляем, удаляем дубликаты
        combined_alts = result["alternative_queries"] + alt_syns
        # Убираем точные дубли, сохраняя порядок
        seen = set()
        deduped = []
        for q in combined_alts:
            if q not in seen:
                seen.add(q)
                deduped.append(q)
        result["alternative_queries"] = deduped

        # Если нет search_keywords, извлечём простые ключевые слова из оригинала
        if not result["search_keywords"]:
            result["search_keywords"] = self._extract_keywords(original_query)

        # Сохраняем итог в metadata, чтобы другие агенты могли использовать
        context.metadata["rewritten_query_result"] = result

        return result

    def _extract_expanded_terms(self, original: str, preprocessed: str) -> Dict[str, str]:
        """
        Проверяем, какие сокращения были расширены: если аббревиатура есть в original, 
        и её полная форма есть в preprocessed, добавляем в expanded_terms.
        """
        expanded: Dict[str, str] = {}
        for abbr, full in self.abbreviations.items():
            # Если аббревиатура встречается как слово в original
            if re.search(r'\b' + re.escape(abbr) + r'\b', original, flags=re.IGNORECASE):
                # И полная форма присутствует в preprocessed
                if re.search(r'\b' + re.escape(full.lower()) + r'\b', preprocessed.lower()):
                    expanded[abbr] = full
        return expanded

    def _generate_alternative_queries(self, query: str) -> List[str]:
        """
        Генерация альтернативных запросов на основе словаря синонимов.
        """
        alternatives: List[str] = []
        q_lower = query.lower()
        for keyword, syns in self.synonyms.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', q_lower):
                for syn in syns:
                    # Заменяем keyword на synonym, сохраняя регистр первой буквы
                    def replace_func(match):
                        word = match.group(0)
                        # Сохраним регистр: если первая буква была заглавной, делаем syn с заглавной
                        if word[0].isupper():
                            return syn.capitalize()
                        else:
                            return syn
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', flags=re.IGNORECASE)
                    alt = pattern.sub(replace_func, query)
                    if alt and alt != query:
                        alternatives.append(alt)
        return alternatives

    def _determine_query_type(self, query: str) -> str:
        """
        Простейшее определение типа запроса по ключевым словам.
        """
        ql = query.lower()
        if any(w in ql for w in ["как", "инструкция", "руководство", "сделать", "настроить"]):
            return "procedural"
        if any(w in ql for w in ["ошибка", "не работает", "проблема", "баг", "исправить"]):
            return "troubleshooting"
        if any(w in ql for w in ["что такое", "описание", "что это", "концепция", "понятие"]):
            return "conceptual"
        return "factual"

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Простая экстракция ключевых слов: разбиваем на слова, удаляем короткие и стоп-слова.
        """
        # Можно расширить стоп-слова, брать из nltk, но здесь простой набор
        stop_words = {"как", "где", "когда", "почему", "что", "это", "на", "в", "и", "или", "не", "для", "по"}
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = []
        for w in words:
            if len(w) <= 2:
                continue
            if w in stop_words:
                continue
            keywords.append(w)
        # Убираем дубликаты, сохраняя порядок
        seen = set()
        uniq = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                uniq.append(w)
        return uniq


def create_query_rewriter_agent(config: Dict[str, Any] = None, langfuse_client=None) -> QueryRewriterAgent:
    """
    Фабричная функция для создания экземпляра QueryRewriterAgent.
    """
    default_config = {
        "model_name": settings.LLM_MODEL_NAME,
        "temperature": 0.1,
        "ollama_base_url": str(settings.LLM_BASE_URL)
    }
    if config:
        default_config.update(config)
    return QueryRewriterAgent(default_config, langfuse_client)
