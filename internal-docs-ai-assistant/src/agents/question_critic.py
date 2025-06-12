import re
from typing import Dict, Any, List, Tuple
import asyncio
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

from base_agent import BaseAgent, AgentContext, with_retry, with_timeout

class QuestionCriticAgent(BaseAgent):
    """Агент для критического анализа и улучшения вопросов"""
    
    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("question_critic", config, langfuse_client)
        
        self.llm = Ollama(
            model=self.get_config("model_name", "llama3.1:8b"),
            temperature=self.get_config("temperature", 0.3),
            base_url=self.get_config("ollama_base_url", "http://localhost:11434")
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template=self._get_prompt_template()
        )
        
        # Предопределенные паттерны для анализа
        self.ambiguity_patterns = [
            r'\b(это|то|такое|подобное)\b',
            r'\b(некоторые|какие-то|всякие)\b',
            r'\b(где-то|как-то|что-то)\b',
        ]
        
        self.grammar_patterns = [
            r'[а-яё]{3,}\s+[а-яё]{3,}\s+[а-яё]{3,}\s+[а-яё]{3,}',
            r'[.!?]{2,}',
            r'\s{2,}',
        ]
        
        # Паттерны для неинформативных запросов
        self._setup_noise_filters()
    
    def _setup_noise_filters(self):
        """Настройка фильтров для неинформативных запросов"""
        
        # Приветствия и вежливые фразы
        self.greeting_patterns = [
            r'\b(привет|здравствуй|добро пожаловать|хай|hello|hi)\b',
            r'\b(как дела|как поживаешь|как жизнь)\b',
            r'\b(добрый (день|утро|вечер))\b',
            r'\b(доброе утро)\b'
        ]
        
        # Благодарности
        self.gratitude_patterns = [
            r'\b(спасибо|благодарю|thanks|thank you)\b',
            r'\b(пасиб|спс|thx)\b'
        ]
        
        # Прощания
        self.farewell_patterns = [
            r'\b(пока|до свидания|goodbye|bye)\b',
            r'\b(увидимся|до встречи)\b'
        ]
        
        # Бессмысленные односложные или короткие фразы
        self.meaningless_patterns = [
            r'^\s*(да|нет|ок|хорошо|плохо|мама|папа|тест|test)\s*$',
            r'^\s*[а-яё]{1,2}\s*$',  # односложные слова
            r'^\s*\d+\s*$',  # только цифры
            r'^\s*[.!?]+\s*$',  # только знаки препинания
        ]
        
        # Эмоциональные выражения без вопроса
        self.emotional_patterns = [
            r'^\s*(ха+|хе+|хи+|лол|ахах|ого|вау|круто)\s*$',
            r'^\s*(ой|ай|эх|ух|ох|ах)\s*$'
        ]
        
        # Тестовые фразы
        self.test_patterns = [
            r'\b(тест|test|проверка|check)\b',
            r'^\s*(123|abc|qwe|йцу)\s*$'
        ]
        
        # Объединяем все паттерны неинформативных запросов
        self.noise_patterns = {
            'greeting': self.greeting_patterns,
            'gratitude': self.gratitude_patterns,
            'farewell': self.farewell_patterns,
            'meaningless': self.meaningless_patterns,
            'emotional': self.emotional_patterns,
            'test': self.test_patterns
        }
    
    def _is_noise_query(self, question: str) -> Tuple[bool, str, str]:
        """
        Проверяет, является ли запрос неинформативным
        
        Returns:
            Tuple[bool, str, str]: (is_noise, noise_type, suggested_response)
        """
        question_clean = question.strip().lower()
        
        if not question_clean or len(question_clean) < 2:
            return True, 'empty', "Пожалуйста, задайте вопрос."
        
        # Проверяем каждую категорию шумовых паттернов
        for noise_type, patterns in self.noise_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_clean, re.IGNORECASE):
                    suggested_response = self._get_noise_response(noise_type, question)
                    return True, noise_type, suggested_response
        
        # Дополнительные эвристики
        words = question_clean.split()
        
        # Слишком короткий запрос без вопросительных слов
        if len(words) == 1 and not any(q_word in question_clean for q_word in 
                                       ['что', 'как', 'где', 'когда', 'почему', 'зачем', 'какой']):
            return True, 'too_short', "Пожалуйста, сформулируйте полный вопрос."
        
        # Запрос состоит только из стоп-слов
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'при', 'через', 'без', 'под'}
        if len(words) <= 3 and all(word in stop_words for word in words):
            return True, 'stop_words', "Пожалуйста, задайте содержательный вопрос."
        
        return False, '', ''
    
    def _get_noise_response(self, noise_type: str, original_question: str) -> str:
        """Генерирует подходящий ответ для неинформативного запроса"""
        
        responses = {
            'greeting': [
                "Привет! Я помогаю находить информацию во внутренней документации. Какой у вас вопрос?",
                "Здравствуйте! Чем могу помочь с поиском информации в документах?",
                "Добро пожаловать! Задайте вопрос по внутренней документации."
            ],
            'gratitude': [
                "Пожалуйста! Если у вас есть еще вопросы по документации, обращайтесь.",
                "Рад помочь! Есть ли еще что-то, что вы хотели бы узнать?",
                "Всегда пожалуйста! Обращайтесь, если понадобится найти что-то в документах."
            ],
            'farewell': [
                "До свидания! Обращайтесь, если понадобится помощь с документацией.",
                "Удачного дня! Всегда готов помочь с поиском информации.",
                "Пока! Возвращайтесь, если будут вопросы по документам."
            ],
            'meaningless': [
                "Я не понял ваш запрос. Пожалуйста, задайте конкретный вопрос по документации.",
                "Не могу обработать такой запрос. Сформулируйте, пожалуйста, вопрос более четко.",
                "Пожалуйста, задайте содержательный вопрос о том, что вы ищете в документах."
            ],
            'emotional': [
                "Понимаю ваши эмоции! Если у вас есть вопрос по документации, я готов помочь.",
                "Есть конкретный вопрос, с которым я могу помочь?",
                "Чем могу быть полезен в поиске информации?"
            ],
            'test': [
                "Система работает корректно. Задайте реальный вопрос по документации для получения помощи.",
                "Тест пройден успешно! Теперь можете задать вопрос по внутренним документам.",
                "Я готов отвечать на вопросы. Что вы хотите найти в документации?"
            ]
        }
        
        import random
        return random.choice(responses.get(noise_type, responses['meaningless']))
    
    def _get_prompt_template(self) -> str:
        """Получение шаблона промпта для анализа вопросов"""
        return """Ты - эксперт по анализу вопросов для системы внутренней документации компании.

Проанализируй следующий вопрос пользователя:
Вопрос: "{question}"

Контекст (если есть): {context}

Выполни анализ по следующим критериям:

1. ЯСНОСТЬ: Насколько понятен вопрос?
2. СПЕЦИФИЧНОСТЬ: Достаточно ли конкретен вопрос?
3. ГРАММАТИКА: Есть ли грамматические ошибки?
4. ПОЛНОТА: Содержит ли вопрос всю необходимую информацию?
5. КОНТЕКСТ: Нужен ли дополнительный контекст?

Предоставь анализ в следующем JSON формате:
{{
  "clarity_score": 0.0-1.0,
  "specificity_score": 0.0-1.0,
  "grammar_score": 0.0-1.0,
  "completeness_score": 0.0-1.0,
  "context_score": 0.0-1.0,
  "overall_score": 0.0-1.0,
  "issues": [
    {{
      "type": "clarity|specificity|grammar|completeness|context",
      "description": "описание проблемы",
      "severity": "low|medium|high",
      "suggestion": "предложение по улучшению"
    }}
  ],
  "improved_question": "улучшенная версия вопроса (если нужно)",
  "clarifying_questions": [
    "вопрос для уточнения 1",
    "вопрос для уточнения 2"
  ],
  "topic_hints": [
    "возможная тема 1",
    "возможная тема 2"
  ]
}}

Отвечай только валидным JSON без дополнительных комментариев."""
    
    @with_timeout(15.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        """Основная логика анализа вопроса"""
        question = context.original_query
        
        # Предварительная проверка на неинформативные запросы
        is_noise, noise_type, suggested_response = self._is_noise_query(question)
        
        if is_noise:
            self.logger.info(f"Обнаружен неинформативный запрос типа '{noise_type}': {question}")
            return {
                "is_noise_query": True,
                "noise_type": noise_type,
                "suggested_response": suggested_response,
                "should_process_further": False,
                "clarity_score": 0.0,
                "specificity_score": 0.0,
                "grammar_score": 1.0,
                "completeness_score": 0.0,
                "context_score": 0.0,
                "overall_score": 0.0,
                "issues": [
                    {
                        "type": "content",
                        "description": f"Неинформативный запрос категории '{noise_type}'",
                        "severity": "high",
                        "suggestion": "Задайте содержательный вопрос по документации"
                    }
                ],
                "improved_question": "Пожалуйста, задайте конкретный вопрос по документации",
                "clarifying_questions": [],
                "topic_hints": [],
                "recommendations": [
                    "Сформулируйте конкретный вопрос",
                    "Укажите, что именно вы ищете в документации",
                    "Добавьте контекст к вашему запросу"
                ]
            }
        
        # Предварительный анализ паттернов для содержательных вопросов
        pattern_analysis = self._analyze_patterns(question)
        
        # Если вопрос проходит базовые проверки, используем LLM
        llm_analysis = await self._llm_analysis(question, context)
        
        # Объединяем результаты
        result = self._combine_analysis(pattern_analysis, llm_analysis)
        
        # Добавляем информацию о том, что это содержательный запрос
        result["is_noise_query"] = False
        result["should_process_further"] = True
        
        # Добавляем рекомендации
        result["recommendations"] = self._generate_recommendations(result)
        
        return result
    
    def _analyze_patterns(self, question: str) -> Dict[str, Any]:
        """Анализ вопроса с помощью паттернов и эвристик"""
        analysis = {
            "length_analysis": self._analyze_length(question),
            "ambiguity_analysis": self._analyze_ambiguity(question),
            "grammar_analysis": self._analyze_grammar(question),
            "question_type": self._determine_question_type(question)
        }
        
        return analysis
    
    def _analyze_length(self, question: str) -> Dict[str, Any]:
        """Анализ длины вопроса"""
        words = question.split()
        chars = len(question)
        
        return {
            "word_count": len(words),
            "char_count": chars,
            "is_too_short": len(words) < 4,
            "is_too_long": len(words) > 50,
            "optimal_length": 6 <= len(words) <= 25
        }
    
    def _analyze_ambiguity(self, question: str) -> Dict[str, Any]:
        """Анализ неоднозначности вопроса"""
        ambiguity_issues = []
        
        for pattern in self.ambiguity_patterns:
            matches = re.findall(pattern, question.lower())
            if matches:
                ambiguity_issues.extend(matches)
        
        return {
            "ambiguous_terms": ambiguity_issues,
            "ambiguity_score": min(len(ambiguity_issues) * 0.2, 1.0),
            "has_ambiguity": len(ambiguity_issues) > 0
        }
    
    def _analyze_grammar(self, question: str) -> Dict[str, Any]:
        """Базовый анализ грамматики"""
        grammar_issues = []
        
        for pattern in self.grammar_patterns:
            if re.search(pattern, question):
                grammar_issues.append(pattern)
        
        starts_with_capital = question[0].isupper() if question else False
        ends_with_question_mark = question.endswith('?') if question else False
        
        return {
            "grammar_issues": grammar_issues,
            "starts_with_capital": starts_with_capital,
            "ends_with_question_mark": ends_with_question_mark,
            "grammar_score": 1.0 - (len(grammar_issues) * 0.2)
        }
    
    def _determine_question_type(self, question: str) -> str:
        """Определение типа вопроса"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['что', 'какой', 'какая', 'какое']):
            return "definition"
        elif any(word in question_lower for word in ['как', 'каким образом']):
            return "instruction"
        elif any(word in question_lower for word in ['где', 'куда']):
            return "location"
        elif any(word in question_lower for word in ['когда', 'во сколько']):
            return "time"
        elif any(word in question_lower for word in ['почему', 'зачем']):
            return "reason"
        elif any(word in question_lower for word in ['можно ли', 'возможно ли']):
            return "possibility"
        else:
            return "general"
    
    async def _llm_analysis(self, question: str, context: AgentContext) -> Dict[str, Any]:
        """Анализ вопроса с помощью LLM (только для содержательных вопросов)"""
        try:
            context_str = ""
            if context.metadata:
                context_str = f"Контекст сессии: {context.metadata}"
            
            prompt = self.prompt_template.format(
                question=question,
                context=context_str
            )
            
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            
            import json
            try:
                llm_result = json.loads(response)
                return llm_result
            except json.JSONDecodeError as e:
                self.logger.warning(f"Не удалось распарсить JSON ответ от LLM: {e}")
                return self._create_fallback_analysis(question)
                
        except Exception as e:
            self.logger.error(f"Ошибка при анализе LLM: {e}")
            return self._create_fallback_analysis(question)
    
    def _create_fallback_analysis(self, question: str) -> Dict[str, Any]:
        """Создание fallback анализа при ошибке LLM"""
        return {
            "clarity_score": 0.7,
            "specificity_score": 0.6,
            "grammar_score": 0.8,
            "completeness_score": 0.6,
            "context_score": 0.5,
            "overall_score": 0.64,
            "issues": [
                {
                    "type": "system",
                    "description": "Анализ выполнен в упрощенном режиме",
                    "severity": "low",
                    "suggestion": "Используется базовый анализ"
                }
            ],
            "improved_question": question,
            "clarifying_questions": [],
            "topic_hints": []
        }
    
    def _combine_analysis(self, pattern_analysis: Dict[str, Any], 
                         llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Объединение результатов паттерн-анализа и LLM-анализа"""
        combined = llm_analysis.copy()
        combined["pattern_analysis"] = pattern_analysis
        
        if pattern_analysis["length_analysis"]["is_too_short"]:
            combined["completeness_score"] = min(combined.get("completeness_score", 0.5), 0.4)
        
        if pattern_analysis["ambiguity_analysis"]["has_ambiguity"]:
            combined["clarity_score"] = min(combined.get("clarity_score", 0.5), 0.6)
        
        scores = [
            combined.get("clarity_score", 0.5),
            combined.get("specificity_score", 0.5),
            combined.get("grammar_score", 0.5),
            combined.get("completeness_score", 0.5),
            combined.get("context_score", 0.5)
        ]
        combined["overall_score"] = sum(scores) / len(scores)
        
        return combined
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по улучшению вопроса"""
        recommendations = []
        
        if analysis.get("clarity_score", 1.0) < 0.7:
            recommendations.append("Сделайте вопрос более четким и понятным")
        
        if analysis.get("specificity_score", 1.0) < 0.7:
            recommendations.append("Добавьте больше конкретных деталей")
        
        if analysis.get("completeness_score", 1.0) < 0.7:
            recommendations.append("Предоставьте больше контекста")
        
        if analysis.get("grammar_score", 1.0) < 0.7:
            recommendations.append("Проверьте грамматику и пунктуацию")
        
        pattern_analysis = analysis.get("pattern_analysis", {})
        if pattern_analysis.get("length_analysis", {}).get("is_too_short"):
            recommendations.append("Расширьте вопрос - он слишком краток")
        
        if pattern_analysis.get("ambiguity_analysis", {}).get("has_ambiguity"):
            recommendations.append("Избегайте неоднозначных формулировок")
        
        return recommendations
    
    def _calculate_confidence(self, result_data: Any, context: AgentContext) -> float:
        """Расчет уверенности в анализе"""
        if not result_data:
            return 0.0
        
        # Для неинформативных запросов уверенность высокая
        if result_data.get("is_noise_query", False):
            return 0.95
        
        overall_score = result_data.get("overall_score", 0.0)
        pattern_analysis = result_data.get("pattern_analysis", {})
        has_pattern_data = bool(pattern_analysis)
        has_llm_data = "issues" in result_data
        
        confidence = overall_score
        
        if has_pattern_data and has_llm_data:
            confidence = min(confidence + 0.1, 1.0)
        
        return confidence

# Фабрика для создания агента
def create_question_critic_agent(config: Dict[str, Any] = None, 
                                langfuse_client=None) -> QuestionCriticAgent:
    """Фабричная функция для создания агента критика вопросов"""
    default_config = {
        "model_name": "llama3.1:8b",
        "temperature": 0.3,
        "ollama_base_url": "http://localhost:11434",
        "max_analysis_time": 15.0,
        "enable_pattern_analysis": True,
        "enable_llm_analysis": True,
        "enable_noise_filtering": True
    }
    
    if config:
        default_config.update(config)
    
    return QuestionCriticAgent(default_config, langfuse_client)