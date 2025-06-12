import re
from typing import Dict, Any, List, Tuple, Optional
import asyncio
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

from agents.base_agent import BaseAgent, AgentContext, with_retry, with_timeout

class TopicClassifierAgent(BaseAgent):
    """Агент для классификации тематики запросов"""
    
    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("topic_classifier", config, langfuse_client)
        
        self.llm = Ollama(
            model=self.get_config("model_name", "llama3.1:8b"),
            temperature=self.get_config("temperature", 0.1),
            base_url=self.get_config("ollama_base_url", "http://localhost:11434")
        )
        
        # Предопределенная иерархия тем
        self.topic_hierarchy = self._setup_topic_hierarchy()
        
        # Ключевые слова для каждой темы
        self.topic_keywords = self._setup_topic_keywords()
        
        # TF-IDF векторизатор для семантического анализа
        self.tfidf_vectorizer = None
        self.topic_vectors = None
        self._initialize_vectorizer()
        
        # Стоп-слова
        try:
            self.stop_words = set(stopwords.words('russian'))
        except:
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('russian'))
        
        # Промпт для LLM классификации
        self.classification_prompt = PromptTemplate(
            input_variables=["question", "topics", "context"],
            template=self._get_classification_prompt()
        )
    
    def _setup_topic_hierarchy(self) -> Dict[str, Dict]:
        """Настройка иерархии тем для внутренней документации"""
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
        """Настройка ключевых слов для каждой темы"""
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
        """Инициализация TF-IDF векторизатора"""
        try:
            # Создаем корпус из ключевых слов каждой темы
            topic_texts = []
            topic_names = []
            
            for topic, keywords in self.topic_keywords.items():
                topic_text = " ".join(keywords)
                topic_texts.append(topic_text)
                topic_names.append(topic)
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=list(self.stop_words),
                ngram_range=(1, 2)
            )
            
            self.topic_vectors = self.tfidf_vectorizer.fit_transform(topic_texts)
            self.topic_names = topic_names
            
        except Exception as e:
            self.logger.warning(f"Не удалось инициализировать TF-IDF: {e}")
            self.tfidf_vectorizer = None
    
    def _get_classification_prompt(self) -> str:
        """Промпт для LLM классификации"""
        return """Ты - эксперт по классификации вопросов для системы внутренней документации компании.

Вопрос пользователя: "{question}"

Контекст (если есть): {context}

Доступные категории:
{topics}

Проанализируй вопрос и определи наиболее подходящую категорию. Учитывай:
1. Основную тематику вопроса
2. Ключевые слова и термины
3. Контекст запроса
4. Возможные подтемы

Ответь в JSON формате:
{{
  "primary_topic": "основная_категория",
  "confidence": 0.0-1.0,
  "secondary_topics": ["дополнительная_категория_1", "дополнительная_категория_2"],
  "reasoning": "объяснение выбора",
  "keywords_found": ["найденные_ключевые_слова"],
  "subtopic_hints": ["возможные_подтемы"]
}}

Отвечай только валидным JSON без дополнительных комментариев."""
    
    @with_timeout(30.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        """Основная логика классификации темы"""
        question = context.original_query
        
        # Многоуровневая классификация
        keyword_analysis = self._analyze_keywords(question)
        tfidf_analysis = self._analyze_tfidf(question)
        llm_analysis = await self._llm_classification(question, context)
        
        # Объединение результатов
        final_classification = self._combine_classifications(
            keyword_analysis, tfidf_analysis, llm_analysis
        )
        
        # Добавление метаданных
        final_classification["analysis_methods"] = {
            "keyword_matching": keyword_analysis is not None,
            "tfidf_similarity": tfidf_analysis is not None,
            "llm_classification": llm_analysis is not None
        }
        
        return final_classification
    
    def _analyze_keywords(self, question: str) -> Dict[str, Any]:
        """Анализ на основе ключевых слов"""
        question_lower = question.lower()
        
        # Токенизация
        try:
            tokens = word_tokenize(question_lower)
            tokens = [token for token in tokens if token not in self.stop_words]
        except:
            tokens = question_lower.split()
        
        # Подсчет совпадений для каждой темы
        topic_scores = {}
        found_keywords = {}
        
        for topic, keywords in self.topic_keywords.items():
            matches = []
            score = 0
            
            for keyword in keywords:
                # Точное совпадение
                if keyword in question_lower:
                    matches.append(keyword)
                    score += 2
                # Частичное совпадение
                elif any(keyword in token for token in tokens):
                    score += 1
            
            topic_scores[topic] = score
            if matches:
                found_keywords[topic] = matches
        
        if not topic_scores or max(topic_scores.values()) == 0:
            return {
                "method": "keyword_matching",
                "primary_topic": "general",
                "confidence": 0.1,
                "topic_scores": topic_scores,
                "found_keywords": found_keywords
            }
        
        # Определение основной темы
        primary_topic = max(topic_scores.items(), key=lambda x: x[1])[0]
        max_score = topic_scores[primary_topic]
        
        # Нормализация уверенности
        total_keywords = sum(len(keywords) for keywords in self.topic_keywords.values())
        confidence = min(max_score / 10, 1.0)  # Масштабирование
        
        # Вторичные темы
        secondary_topics = [
            topic for topic, score in sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[1:3]
            if score > 0
        ]
        
        return {
            "method": "keyword_matching",
            "primary_topic": primary_topic,
            "confidence": confidence,
            "secondary_topics": secondary_topics,
            "topic_scores": topic_scores,
            "found_keywords": found_keywords
        }
    
    def _analyze_tfidf(self, question: str) -> Optional[Dict[str, Any]]:
        """Анализ на основе TF-IDF сходства"""
        if not self.tfidf_vectorizer:
            return None
        
        try:
            # Векторизация вопроса
            question_vector = self.tfidf_vectorizer.transform([question.lower()])
            
            # Вычисление сходства с каждой темой
            similarities = cosine_similarity(question_vector, self.topic_vectors)[0]
            
            # Сортировка по убыванию сходства
            topic_similarities = list(zip(self.topic_names, similarities))
            topic_similarities.sort(key=lambda x: x[1], reverse=True)
            
            primary_topic = topic_similarities[0][0]
            confidence = topic_similarities[0][1]
            
            # Вторичные темы с достаточным сходством
            secondary_topics = [
                topic for topic, sim in topic_similarities[1:3]
                if sim > 0.1
            ]
            
            return {
                "method": "tfidf_similarity",
                "primary_topic": primary_topic,
                "confidence": float(confidence),
                "secondary_topics": secondary_topics,
                "similarities": dict(topic_similarities)
            }
            
        except Exception as e:
            self.logger.warning(f"Ошибка TF-IDF анализа: {e}")
            return None
    
    async def _llm_classification(self, question: str, context: AgentContext) -> Optional[Dict[str, Any]]:
        """Классификация с помощью LLM"""
        try:
            # Подготовка описания тем
            topics_description = []
            for topic_id, topic_info in self.topic_hierarchy.items():
                topics_description.append(
                    f"- {topic_id}: {topic_info['name']} (подтемы: {', '.join(topic_info['subtopics'][:5])})"
                )
            
            context_str = ""
            if context.metadata:
                context_str = str(context.metadata)
            
            prompt = self.classification_prompt.format(
                question=question,
                topics="\n".join(topics_description),
                context=context_str
            )
            
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            
            import json
            try:
                llm_result = json.loads(response)
                llm_result["method"] = "llm_classification"
                return llm_result
            except json.JSONDecodeError:
                self.logger.warning("Не удалось распарсить JSON ответ от LLM")
                return None
                
        except Exception as e:
            self.logger.error(f"Ошибка LLM классификации: {e}")
            return None
    
    def _combine_classifications(self, keyword_analysis: Dict, 
                                tfidf_analysis: Optional[Dict], 
                                llm_analysis: Optional[Dict]) -> Dict[str, Any]:
        """Объединение результатов разных методов классификации"""
        
        # Веса для разных методов
        weights = {
            "keyword_matching": 0.4,
            "tfidf_similarity": 0.3,
            "llm_classification": 0.3
        }
        
        # Сбор всех предложенных тем
        topic_votes = {}
        analysis_details = {}
        
        # Обработка результатов ключевых слов
        if keyword_analysis:
            primary = keyword_analysis["primary_topic"]
            confidence = keyword_analysis["confidence"]
            
            topic_votes[primary] = topic_votes.get(primary, 0) + weights["keyword_matching"] * confidence
            analysis_details["keyword_analysis"] = keyword_analysis
        
        # Обработка TF-IDF результатов
        if tfidf_analysis:
            primary = tfidf_analysis["primary_topic"]
            confidence = tfidf_analysis["confidence"]
            
            topic_votes[primary] = topic_votes.get(primary, 0) + weights["tfidf_similarity"] * confidence
            analysis_details["tfidf_analysis"] = tfidf_analysis
        
        # Обработка LLM результатов
        if llm_analysis:
            primary = llm_analysis["primary_topic"]
            confidence = llm_analysis.get("confidence", 0.5)
            
            topic_votes[primary] = topic_votes.get(primary, 0) + weights["llm_classification"] * confidence
            analysis_details["llm_analysis"] = llm_analysis
        
        # Определение финальной темы
        if not topic_votes:
            final_topic = "general"
            final_confidence = 0.1
        else:
            final_topic = max(topic_votes.items(), key=lambda x: x[1])[0]
            final_confidence = min(topic_votes[final_topic], 1.0)
        
        # Сбор вторичных тем
        secondary_topics = []
        all_secondary = set()
        
        for analysis in [keyword_analysis, tfidf_analysis, llm_analysis]:
            if analysis and "secondary_topics" in analysis:
                all_secondary.update(analysis["secondary_topics"])
        
        secondary_topics = [topic for topic in all_secondary if topic != final_topic][:3]
        
        # Подтемы и ключевые слова
        subtopic_hints = []
        keywords_found = []
        
        if final_topic in self.topic_hierarchy:
            subtopic_hints = self.topic_hierarchy[final_topic]["subtopics"][:5]
        
        if keyword_analysis and "found_keywords" in keyword_analysis:
            keywords_found = keyword_analysis["found_keywords"].get(final_topic, [])
        
        # Определение приоритета
        priority = self.topic_hierarchy.get(final_topic, {}).get("priority", 4)
        
        return {
            "primary_topic": final_topic,
            "confidence": final_confidence,
            "secondary_topics": secondary_topics,
            "priority": priority,
            "topic_name": self.topic_hierarchy.get(final_topic, {}).get("name", final_topic),
            "subtopic_hints": subtopic_hints,
            "keywords_found": keywords_found,
            "topic_votes": topic_votes,
            "analysis_details": analysis_details,
            "reasoning": f"Классификация на основе {len([a for a in [keyword_analysis, tfidf_analysis, llm_analysis] if a])} методов"
        }
    
    def _calculate_confidence(self, result_data: Any, context: AgentContext) -> float:
        """Расчет уверенности в классификации"""
        if not result_data:
            return 0.0
        
        base_confidence = result_data.get("confidence", 0.0)
        
        # Бонус за согласованность методов
        analysis_details = result_data.get("analysis_details", {})
        methods_count = len(analysis_details)
        
        if methods_count > 1:
            # Проверка согласованности
            primary_topics = set()
            for analysis in analysis_details.values():
                if "primary_topic" in analysis:
                    primary_topics.add(analysis["primary_topic"])
            
            if len(primary_topics) == 1:  # Все методы согласны
                base_confidence = min(base_confidence + 0.2, 1.0)
            elif len(primary_topics) <= 2:  # Частичное согласие
                base_confidence = min(base_confidence + 0.1, 1.0)
        
        # Учет приоритета темы
        priority = result_data.get("priority", 4)
        priority_bonus = (5 - priority) * 0.05
        
        return min(base_confidence + priority_bonus, 1.0)

# Фабрика для создания агента
def create_topic_classifier_agent(config: Dict[str, Any] = None, 
                                 langfuse_client=None) -> TopicClassifierAgent:
    """Фабричная функция для создания агента классификации тем"""
    default_config = {
        "model_name": "llama3.1:8b",
        "temperature": 0.1,
        "ollama_base_url": "http://localhost:11434",
        "use_tfidf": True,
        "use_keywords": True,
        "use_llm": True,
        "min_confidence_threshold": 0.3
    }
    
    if config:
        default_config.update(config)
    
    return TopicClassifierAgent(default_config, langfuse_client)