import hashlib
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import redis.asyncio as aioredis

from agents.base_agent import BaseAgent, AgentContext, with_retry, with_timeout
# Используйте правильный импорт Ollama в вашей среде:
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from configs.settings import settings  # скорректируйте путь, если нужно


class TopicClassifierAgent(BaseAgent):
    """Агент для классификации тематики запросов"""

    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("topic_classifier", config, langfuse_client)
        # Настройка LLM через Ollama
        model_name = self.get_config("model_name", settings.LLM_MODEL_NAME)
        base_url = self.get_config("ollama_base_url", None) or str(settings.LLM_BASE_URL)
        temperature = float(self.get_config("temperature", 0.1))
        self.llm = Ollama(model=model_name, temperature=temperature, base_url=base_url)

        # Подключение к Redis для кэша классификации
        redis_url = getattr(settings, "REDIS_URL", None)
        if redis_url:
            try:
                self.redis = aioredis.from_url(redis_url)
            except Exception as e:
                self.logger.warning(f"TopicClassifierAgent: не удалось подключиться к Redis: {e}")
                self.redis = None
        else:
            self.redis = None

        # Иерархия тем и ключевые слова
        # Можно переопределить через config: config может содержать "topic_hierarchy" и "topic_keywords"
        self.topic_hierarchy = config.get("topic_hierarchy", self._setup_topic_hierarchy())
        self.topic_keywords = config.get("topic_keywords", self._setup_topic_keywords())

        # TF-IDF векторизатор
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.topic_vectors = None
        self.topic_names: List[str] = []
        self._initialize_vectorizer()

        # Stop-слова для русского (или других языков, в зависимости от задачи)
        try:
            # проверим, загружены ли корпуса
            _ = stopwords.words('russian')
        except (LookupError, OSError):
            nltk.download('stopwords')
            nltk.download('punkt')
        try:
            self.stop_words = set(stopwords.words('russian'))
        except Exception:
            # Если по какой-то причине не удалось: пустой набор
            self.stop_words = set()

        # PromptTemplate для LLM-классификации
        self.classification_prompt = PromptTemplate(
            input_variables=["question", "topics", "context_str"],
            template=self._get_classification_prompt()
        )

        # Логгер уровень DEBUG для подробностей
        self.logger.setLevel(logging.DEBUG)

    def _setup_topic_hierarchy(self) -> Dict[str, Dict]:
        """Дефолтная иерархия тем (пример)."""
        return {
            "hr": {
                "name": "HR и кадровые вопросы",
                "subtopics": [
                    "отпуска", "больничные", "зарплата", "премии",
                    "найм", "увольнение", "аттестация", "обучение",
                    "корпоративные льготы", "дресс-код"
                ],
                "priority": 1
            },
            "it": {
                "name": "IT и техническая поддержка",
                "subtopics": [
                    "программное обеспечение", "оборудование", "сеть",
                    "безопасность", "доступы", "антивирус", "backup",
                    "печать", "техподдержка", "системы"
                ],
                "priority": 2
            },
            "finance": {
                "name": "Финансы и бухгалтерия",
                "subtopics": [
                    "отчетность", "бюджет", "расходы", "договоры",
                    "налоги", "командировочные", "закупки", "оплата"
                ],
                "priority": 1
            },
            "legal": {
                "name": "Юридические вопросы",
                "subtopics": [
                    "договоры", "соглашения", "регламенты", "политики",
                    "конфиденциальность", "персональные данные", "лицензии"
                ],
                "priority": 2
            },
            "security": {
                "name": "Безопасность",
                "subtopics": [
                    "информационная безопасность", "физическая безопасность",
                    "пропуска", "видеонаблюдение", "инциденты", "нарушения"
                ],
                "priority": 1
            },
            "facilities": {
                "name": "Административно-хозяйственные вопросы",
                "subtopics": [
                    "офис", "уборка", "ремонт", "мебель", "канцелярия",
                    "питание", "парковка", "переговорные"
                ],
                "priority": 3
            },
            "processes": {
                "name": "Бизнес-процессы",
                "subtopics": [
                    "регламенты", "процедуры", "документооборот",
                    "согласования", "подписи", "workflow"
                ],
                "priority": 2
            },
            "management": {
                "name": "Управленческие вопросы",
                "subtopics": [
                    "стратегия", "планирование", "встречи", "отчеты",
                    "KPI", "цели", "проекты", "решения"
                ],
                "priority": 2
            },
            "compliance": {
                "name": "Соответствие и аудит",
                "subtopics": [
                    "аудит", "соответствие", "стандарты", "сертификация",
                    "проверки", "требования", "нормативы"
                ],
                "priority": 3
            },
            "general": {
                "name": "Общие вопросы",
                "subtopics": [
                    "новости", "объявления", "события", "контакты",
                    "структура", "информация"
                ],
                "priority": 4
            }
        }

    def _setup_topic_keywords(self) -> Dict[str, List[str]]:
        """Дефолтные ключевые слова для каждой темы."""
        return {
            "hr": [
                "отпуск", "больничный", "зарплата", "оклад", "премия", "найм", "увольнение",
                "трудовой", "договор", "кадры", "персонал", "сотрудник", "работник",
                "аттестация", "обучение", "курсы", "льготы", "соцпакет", "дресс-код",
                "рабочее время", "график", "переработка", "отгул"
            ],
            "it": [
                "компьютер", "ноутбук", "программа", "софт", "сеть", "интернет",
                "доступ", "пароль", "логин", "антивирус", "backup", "резервная копия",
                "принтер", "печать", "сканер", "техподдержка", "система", "сервер",
                "базы данных", "1С", "CRM", "ERP", "лицензия"
            ],
            "finance": [
                "деньги", "бюджет", "расходы", "доходы", "отчет", "бухгалтерия",
                "налог", "НДС", "договор", "счет", "оплата", "командировка",
                "авансовый отчет", "закупка", "тендер", "поставщик", "подрядчик"
            ],
            "legal": [
                "договор", "соглашение", "право", "закон", "юрист", "юридический",
                "регламент", "политика", "процедура", "конфиденциальность",
                "персональные данные", "GDPR", "лицензия", "патент", "иск"
            ],
            "security": [
                "безопасность", "пропуск", "доступ", "охрана", "видеонаблюдение",
                "инцидент", "нарушение", "угроза", "защита", "контроль",
                "информационная безопасность", "физическая безопасность"
            ],
            "facilities": [
                "офис", "здание", "помещение", "уборка", "клининг", "ремонт",
                "мебель", "канцелярия", "питание", "столовая", "кафе",
                "парковка", "переговорная", "зал", "кондиционер", "отопление"
            ],
            "processes": [
                "процесс", "процедура", "регламент", "инструкция", "порядок",
                "документооборот", "согласование", "подпись", "утверждение",
                "workflow", "алгоритм", "схема", "этап"
            ],
            "management": [
                "управление", "менеджмент", "руководство", "директор", "начальник",
                "стратегия", "план", "планирование", "встреча", "совещание",
                "отчет", "KPI", "показатели", "цель", "задача", "проект"
            ],
            "compliance": [
                "аудит", "проверка", "соответствие", "стандарт", "ISO", "сертификат",
                "требование", "норматив", "контроль качества", "аккредитация"
            ],
            "general": [
                "информация", "новости", "объявление", "событие", "контакт",
                "телефон", "адрес", "структура", "организация", "подразделение"
            ]
        }

    def _initialize_vectorizer(self):
        """Инициализация TF-IDF векторизатора на корпусе keyword lists."""
        try:
            topic_texts = []
            topic_names = []
            for topic, keywords in self.topic_keywords.items():
                if isinstance(keywords, list) and keywords:
                    topic_text = " ".join(keywords)
                else:
                    topic_text = ""
                topic_texts.append(topic_text)
                topic_names.append(topic)
            # Инициализируем vectorizer, исключая стоп-слова
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=list(self.stop_words) if self.stop_words else None,
                ngram_range=(1, 2)
            )
            self.topic_vectors = self.tfidf_vectorizer.fit_transform(topic_texts)
            self.topic_names = topic_names
        except Exception as e:
            self.logger.warning(f"TopicClassifierAgent: не удалось инициализировать TF-IDF: {e}")
            self.tfidf_vectorizer = None
            self.topic_vectors = None
            self.topic_names = []

    def _get_classification_prompt(self) -> str:
        """PromptTemplate для LLM-классификации."""
        return """Ты - эксперт по классификации вопросов для системы внутренней документации компании.

Вопрос пользователя: "{question}"

Контекст (если есть): {context_str}

Доступные категории:
{topics}

Проанализируй вопрос и определи наиболее подходящую категорию. Учитывай:
1. Основную тематику вопроса
2. Ключевые слова и термины
3. Контекст запроса
4. Возможные подтемы

Ответь строго в JSON формате:
{{
  "primary_topic": "основная_категория", 
  "confidence": 0.0-1.0, 
  "secondary_topics": ["дополнительная_категория_1", "дополнительная_категория_2"], 
  "reasoning": "объяснение выбора", 
  "keywords_found": ["найденные_ключевые_слова"], 
  "subtopic_hints": ["возможные_подтемы"]
}}

Отвечай только валидным JSON без лишних комментариев."""

    @with_timeout(30.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        """Основная логика классификации темы."""
        question = context.processed_query or context.original_query or ""
        question_str = question.strip()
        if not question_str:
            # Пустой или отсутствующий запрос
            result = {
                "primary_topic": "general",
                "confidence": 0.0,
                "secondary_topics": [],
                "reasoning": "Нет текста запроса для классификации",
                "keywords_found": [],
                "subtopic_hints": []
            }
            # Сохраняем в metadata
            context.metadata["topic_classification"] = result
            return result

        # Кэширование: вычисляем ключ по хэшу вопроса
        cache_key = None
        if self.redis:
            # используем sha256, чтобы ключ был фиксированной длины
            h = hashlib.sha256(question_str.encode('utf-8')).hexdigest()
            cache_key = f"topic_classify:{h}"
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    try:
                        cached_obj = json.loads(cached)
                        self.logger.debug("TopicClassifierAgent: взят результат из кеша")
                        # Сохраняем в metadata и возвращаем
                        context.metadata["topic_classification"] = cached_obj
                        return cached_obj
                    except Exception:
                        pass
            except Exception as e:
                self.logger.warning(f"TopicClassifierAgent: не удалось прочитать из Redis: {e}")

        # Keyword-анализ
        keyword_analysis = self._analyze_keywords(question_str)
        # TF-IDF-анализ
        tfidf_analysis = self._analyze_tfidf(question_str)
        # LLM-анализ
        llm_analysis = await self._llm_classification(question_str, context)

        # Объединение
        final_cls = self._combine_classifications(keyword_analysis, tfidf_analysis, llm_analysis)

        # Добавляем детали методов
        final_cls["analysis_methods"] = {
            "keyword_matching": keyword_analysis is not None,
            "tfidf_similarity": tfidf_analysis is not None,
            "llm_classification": llm_analysis is not None
        }

        # Сохраняем в metadata
        context.metadata["topic_classification"] = final_cls

        # Сохраняем в Redis-кэш с TTL (например, 3600 сек), если удалось соединение
        if self.redis and cache_key:
            try:
                await self.redis.set(cache_key, json.dumps(final_cls, ensure_ascii=False), ex=3600)
            except Exception as e:
                self.logger.warning(f"TopicClassifierAgent: не удалось записать в Redis: {e}")

        return final_cls

    def _analyze_keywords(self, question: str) -> Dict[str, Any]:
        """Анализ на основе ключевых слов."""
        q_lower = question.lower()
        # Токенизация
        try:
            tokens = word_tokenize(q_lower)
            tokens = [t for t in tokens if t not in self.stop_words]
        except Exception:
            tokens = q_lower.split()

        topic_scores: Dict[str, int] = {}
        found_keywords: Dict[str, List[str]] = {}
        for topic, keywords in self.topic_keywords.items():
            score = 0
            matches: List[str] = []
            for kw in keywords:
                if not kw:
                    continue
                # точное вхождение
                if kw.lower() in q_lower:
                    score += 2
                    matches.append(kw)
                else:
                    # частичное сравнение по токенам
                    for token in tokens:
                        if kw.lower() in token:
                            score += 1
                            break
            topic_scores[topic] = score
            if matches:
                found_keywords[topic] = matches

        # Если нет совпадений
        if not topic_scores or max(topic_scores.values()) == 0:
            return {
                "method": "keyword_matching",
                "primary_topic": "general",
                "confidence": 0.1,
                "secondary_topics": [],
                "topic_scores": topic_scores,
                "found_keywords": found_keywords
            }

        # Выбираем основной по максимальному score
        primary, max_score = max(topic_scores.items(), key=lambda x: x[1])
        # Нормализация confidence: например, делим на некоторый порог, можно настроить
        confidence = min(max_score / 10.0, 1.0)

        # Вторичные темы: топ-2 после основной
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        secondary = []
        for t, s in sorted_topics[1:3]:
            if s > 0:
                secondary.append(t)
        return {
            "method": "keyword_matching",
            "primary_topic": primary,
            "confidence": confidence,
            "secondary_topics": secondary,
            "topic_scores": topic_scores,
            "found_keywords": found_keywords
        }

    def _analyze_tfidf(self, question: str) -> Optional[Dict[str, Any]]:
        """Анализ на основе TF-IDF сходства с keyword corpus."""
        if not self.tfidf_vectorizer or self.topic_vectors is None:
            return None
        try:
            q_vec = self.tfidf_vectorizer.transform([question.lower()])
            sims = cosine_similarity(q_vec, self.topic_vectors)[0]  # массив длины len(topic_names)
            topic_sims = list(zip(self.topic_names, sims))
            # Сортировка по убыванию сходства
            topic_sims.sort(key=lambda x: x[1], reverse=True)
            primary, conf = topic_sims[0]
            # Вторичные с порогом
            secondary = [t for t, sim in topic_sims[1:3] if sim > 0.1]
            # Словарь всех similarities для отладки, но может быть большим
            sims_dict = {t: float(sim) for t, sim in topic_sims}
            return {
                "method": "tfidf_similarity",
                "primary_topic": primary,
                "confidence": float(conf),
                "secondary_topics": secondary,
                "similarities": sims_dict
            }
        except Exception as e:
            self.logger.warning(f"TopicClassifierAgent: ошибка TF-IDF анализа: {e}")
            return None

    async def _llm_classification(self, question: str, context: AgentContext) -> Optional[Dict[str, Any]]:
        """Классификация с помощью LLM."""
        try:
            # Формируем описание тем
            topics_desc_lines: List[str] = []
            for topic_id, info in self.topic_hierarchy.items():
                name = info.get("name", topic_id)
                subs = info.get("subtopics", [])
                # Ограничиваем субтемы первым 5 для краткости
                subs_str = ", ".join(subs[:5]) if isinstance(subs, list) else ""
                topics_desc_lines.append(f"- {topic_id}: {name} (подтемы: {subs_str})")
            topics_str = "\n".join(topics_desc_lines)

            # Контекст: можно включить topic hints из предыдущей классификации или metadata
            context_str = ""
            if context.metadata:
                prev = context.metadata.get("topic_classification")
                if isinstance(prev, dict):
                    # Можно включить прошлый результат кратко
                    prev_primary = prev.get("primary_topic")
                    context_str = f"Предыдущее определение темы: {prev_primary}"
                else:
                    # либо широкое содержание metadata
                    context_str = json.dumps(context.metadata, ensure_ascii=False)
            prompt = self.classification_prompt.format(
                question=question,
                topics=topics_str,
                context_str=context_str or "нет"
            )
            # Вызываем LLM через invoke_llm
            response = await self.invoke_llm(prompt)
        except Exception as e:
            self.logger.error(f"TopicClassifierAgent: ошибка при вызове LLM: {e}")
            return None

        parsed = self.parse_json_response(response.strip())
        if not isinstance(parsed, dict):
            self.logger.warning("TopicClassifierAgent: LLM вернул не JSON-объект")
            return None

        # Валидация структуры: ожидаемые поля: primary_topic, confidence, secondary_topics, reasoning, keywords_found, subtopic_hints
        validated = {}
        # primary_topic
        pt = parsed.get("primary_topic")
        if isinstance(pt, str) and pt in self.topic_hierarchy:
            validated["primary_topic"] = pt
        else:
            # если неверно или отсутствует, оставляем None, объединение заполнит потом
            validated["primary_topic"] = None
        # confidence
        conf = parsed.get("confidence")
        if isinstance(conf, (int, float)):
            try:
                conf_f = float(conf)
                validated["confidence"] = max(0.0, min(1.0, conf_f))
            except:
                validated["confidence"] = None
        else:
            validated["confidence"] = None
        # secondary_topics
        st = parsed.get("secondary_topics")
        if isinstance(st, list):
            # фильтруем строки, оставляем только валидные ключи
            cleaned = [str(x) for x in st if isinstance(x, str) and x in self.topic_hierarchy and x != validated.get("primary_topic")]
            validated["secondary_topics"] = cleaned[:3]
        else:
            validated["secondary_topics"] = []
        # reasoning
        reasoning = parsed.get("reasoning")
        if isinstance(reasoning, str):
            validated["reasoning"] = reasoning.strip()
        else:
            validated["reasoning"] = ""
        # keywords_found
        kf = parsed.get("keywords_found")
        if isinstance(kf, list):
            # строки
            validated["keywords_found"] = [str(x) for x in kf if isinstance(x, str)]
        else:
            validated["keywords_found"] = []
        # subtopic_hints
        sh = parsed.get("subtopic_hints")
        if isinstance(sh, list):
            validated["subtopic_hints"] = [str(x) for x in sh if isinstance(x, str)]
        else:
            validated["subtopic_hints"] = []

        validated["method"] = "llm_classification"
        return validated

    def _combine_classifications(
        self,
        keyword_analysis: Dict[str, Any],
        tfidf_analysis: Optional[Dict[str, Any]],
        llm_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Объединение результатов keyword-, TF-IDF- и LLM-анализов.
        Возвращает финальную структуру с primary_topic, confidence, secondary_topics, reasoning, keywords_found, subtopic_hints, priority и topic_name.
        """
        # Веса можно брать из config или фиксировать
        weights = {
            "keyword_matching": float(self.get_config("weight_keyword", 0.4)),
            "tfidf_similarity": float(self.get_config("weight_tfidf", 0.3)),
            "llm_classification": float(self.get_config("weight_llm", 0.3))
        }
        topic_votes: Dict[str, float] = {}
        analysis_details: Dict[str, Any] = {}

        # keyword
        if keyword_analysis:
            pt = keyword_analysis.get("primary_topic")
            conf = keyword_analysis.get("confidence", 0.0)
            if pt:
                topic_votes[pt] = topic_votes.get(pt, 0.0) + weights["keyword_matching"] * conf
            analysis_details["keyword_analysis"] = keyword_analysis

        # tfidf
        if tfidf_analysis:
            pt = tfidf_analysis.get("primary_topic")
            conf = tfidf_analysis.get("confidence", 0.0)
            if pt:
                topic_votes[pt] = topic_votes.get(pt, 0.0) + weights["tfidf_similarity"] * conf
            analysis_details["tfidf_analysis"] = tfidf_analysis

        # llm
        if llm_analysis:
            pt = llm_analysis.get("primary_topic")
            conf = llm_analysis.get("confidence", 0.5) if llm_analysis.get("confidence") is not None else 0.5
            if pt:
                topic_votes[pt] = topic_votes.get(pt, 0.0) + weights["llm_classification"] * conf
            analysis_details["llm_analysis"] = llm_analysis

        # Если нет голосов, default to general
        if not topic_votes:
            final_topic = "general"
            final_confidence = 0.1
        else:
            # Выбираем topic с max vote
            final_topic, vote_score = max(topic_votes.items(), key=lambda x: x[1])
            final_confidence = min(vote_score, 1.0)

        # Secondary topics: собрать из всех анализов, исключив финальную тему
        secondary_set = set()
        for analysis in (keyword_analysis, tfidf_analysis, llm_analysis):
            if isinstance(analysis, dict):
                secs = analysis.get("secondary_topics") or []
                for t in secs:
                    if t != final_topic:
                        secondary_set.add(t)
        secondary_topics = list(secondary_set)[:3]

        # Subtopic hints: по иерархии
        subtopic_hints = []
        if final_topic in self.topic_hierarchy:
            subs = self.topic_hierarchy[final_topic].get("subtopics", [])
            if isinstance(subs, list):
                subtopic_hints = subs[:5]

        # Keywords found: из keyword_analysis для финальной темы
        keywords_found = []
        if keyword_analysis:
            found = keyword_analysis.get("found_keywords", {}).get(final_topic, [])
            if isinstance(found, list):
                keywords_found = [str(x) for x in found]

        # Reasoning: объединяем reasoning из LLM, либо синтезируем
        reasoning = ""
        if llm_analysis and llm_analysis.get("reasoning"):
            reasoning = llm_analysis["reasoning"]
        else:
            # Можно сформировать простое reasoning
            reasoning = f"Выбрана тема '{final_topic}' на основе комбинации keyword- и TF-IDF-анализа"
        # Priority & topic_name
        priority = self.topic_hierarchy.get(final_topic, {}).get("priority", 5)
        topic_name = self.topic_hierarchy.get(final_topic, {}).get("name", final_topic)

        return {
            "primary_topic": final_topic,
            "confidence": final_confidence,
            "secondary_topics": secondary_topics,
            "reasoning": reasoning,
            "keywords_found": keywords_found,
            "subtopic_hints": subtopic_hints,
            "topic_name": topic_name,
            "priority": priority,
            "topic_votes": topic_votes,
            "analysis_details": analysis_details
        }

    def _calculate_confidence(self, result_data: Any, context: AgentContext) -> float:
        """Расчёт уверенности: используем result_data["confidence"], с бонусом, если методы согласуются."""
        if not isinstance(result_data, dict):
            return 0.0
        base = result_data.get("confidence", 0.0)
        details = result_data.get("analysis_details", {})
        # бонус за согласованность: если keyword и tfidf и llm выбрали одну тему, +0.1
        topics = []
        for m in ("keyword_analysis", "tfidf_analysis", "llm_analysis"):
            part = details.get(m)
            if isinstance(part, dict) and part.get("primary_topic"):
                topics.append(part.get("primary_topic"))
        if topics and len(set(topics)) == 1:
            base = min(base + 0.1, 1.0)
        return max(0.0, min(1.0, float(base)))

    async def _postprocess(self, result_data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """
        Сохраняем результат в metadata, чтобы другие агенты могли использовать:
        context.metadata['topic_classification'] = result_data
        """
        context.metadata["topic_classification"] = result_data
        return result_data


def create_topic_classifier_agent(config: Dict[str, Any] = None, langfuse_client=None) -> TopicClassifierAgent:
    """
    Фабричная функция для создания TopicClassifierAgent.
    config может содержать:
      - model_name, temperature, ollama_base_url
      - topic_hierarchy (dict), topic_keywords (dict)
      - weight_keyword, weight_tfidf, weight_llm
    """
    default_config = {
        "model_name": settings.LLM_MODEL_NAME,
        "temperature": 0.1,
        "ollama_base_url": str(settings.LLM_BASE_URL),
        # веса для объединения
        "weight_keyword": 0.4,
        "weight_tfidf": 0.3,
        "weight_llm": 0.3,
        # topic_hierarchy и topic_keywords при отсутствии будут использованы дефолтные из класса
    }
    if config:
        default_config.update(config)
    return TopicClassifierAgent(default_config, langfuse_client)
