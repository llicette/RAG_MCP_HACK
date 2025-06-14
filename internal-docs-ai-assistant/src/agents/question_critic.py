import re
import json
import random
from typing import Dict, Any, List, Tuple
import asyncio

from agents.base_agent import BaseAgent, AgentContext, with_retry, with_timeout
# Используйте правильный импорт Ollama в вашем окружении:
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from configs.settings import settings  # скорректируйте путь при необходимости


class QuestionCriticAgent(BaseAgent):
    """Агент для критического анализа и улучшения вопросов."""

    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("question_critic", config, langfuse_client)

        # Инициализация LLM с параметрами из конфига или settings
        model_name = self.get_config("model_name", settings.LLM_MODEL_NAME)
        base_url = self.get_config("ollama_base_url", None) or str(settings.LLM_BASE_URL)
        temperature = float(self.get_config("temperature", 0.3))
        self.llm = Ollama(model=model_name, temperature=temperature, base_url=base_url)

        # PromptTemplate: экранируем JSON-образец двойными {{ }}
        self.prompt_template = PromptTemplate(
            input_variables=["question", "context_str"],
            template=self._get_prompt_template()
        )

        # Предопределённые паттерны для анализа
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

        # Шумовые фильтры
        self._setup_noise_filters()

    def _setup_noise_filters(self):
        """Настройка фильтров для неинформативных запросов."""
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
        # Бессмысленные фразы
        self.meaningless_patterns = [
            r'^\s*(да|нет|ок|хорошо|плохо|мама|папа|тест|test)\s*$',
            r'^\s*[а-яё]{1,2}\s*$',  # односложные слова
            r'^\s*\d+\s*$',         # только цифры
            r'^\s*[.!?]+\s*$',      # только знаки препинания
        ]
        # Эмоции
        self.emotional_patterns = [
            r'^\s*(ха+|хе+|хи+|лол|ахах|ого|вау|круто)\s*$',
            r'^\s*(ой|ай|эх|ух|ох|ах)\s*$'
        ]
        # Тестовые фразы
        self.test_patterns = [
            r'\b(тест|test|проверка|check)\b',
            r'^\s*(123|abc|qwe|йцу)\s*$'
        ]
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
        Проверяет, является ли запрос неинформативным.
        Returns: (is_noise, noise_type, suggested_response)
        """
        question_clean = (question or "").strip().lower()
        if not question_clean or len(question_clean) < 2:
            return True, 'empty', "Пожалуйста, задайте вопрос."

        # Шумовые паттерны
        for noise_type, patterns in self.noise_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_clean, re.IGNORECASE):
                    suggested_response = self._get_noise_response(noise_type)
                    return True, noise_type, suggested_response

        # Слишком короткий запрос без вопросительных слов
        words = question_clean.split()
        if len(words) == 1 and not any(qw in question_clean for qw in 
                                       ['что', 'как', 'где', 'когда', 'почему', 'зачем', 'какой']):
            return True, 'too_short', "Пожалуйста, сформулируйте полный вопрос."
        # Запрос состоит только из стоп-слов
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'при', 'через', 'без', 'под'}
        if len(words) <= 3 and all(word in stop_words for word in words):
            return True, 'stop_words', "Пожалуйста, задайте содержательный вопрос."
        return False, '', ''

    def _get_noise_response(self, noise_type: str) -> str:
        """Генерирует ответ-подсказку для шумовых запросов."""
        responses = {
            'greeting': [
                "Привет! Я помогаю находить информацию во внутренней документации. Какой у вас вопрос?",
                "Здравствуйте! Чем могу помочь с поиском информации в документах?",
                "Добро пожаловать! Задайте вопрос по внутренней документации."
            ],
            'gratitude': [
                "Пожалуйста! Если у вас есть ещё вопросы по документации, обращайтесь.",
                "Рад помочь! Есть ли ещё что-то, что вы хотели бы узнать?",
                "Всегда пожалуйста! Обращайтесь за помощью."
            ],
            'farewell': [
                "До свидания! Обращайтесь, если понадобится помощь.",
                "Удачного дня! Всегда готов помочь с документацией.",
                "Пока! Возвращайтесь, если будут вопросы."
            ],
            'meaningless': [
                "Я не понял ваш запрос. Пожалуйста, задайте конкретный вопрос по документации.",
                "Не могу обработать такой запрос. Сформулируйте, пожалуйста, более чётко.",
                "Пожалуйста, задайте содержательный вопрос."
            ],
            'emotional': [
                "Понимаю ваши эмоции! Если есть вопрос по документации, я готов помочь.",
                "Есть конкретный вопрос, с которым я могу помочь?",
                "Чем могу быть полезен в поиске информации?"
            ],
            'test': [
                "Система работает корректно. Задайте реальный вопрос по документации.",
                "Тест успешен! Теперь задайте вопрос по внутренним документам.",
                "Я готов отвечать. Что хотите найти в документации?"
            ],
            'empty': [
                "Пожалуйста, задайте вопрос.",
            ],
            'too_short': [
                "Пожалуйста, сформулируйте более развёрнутый вопрос.",
            ],
            'stop_words': [
                "Пожалуйста, задайте содержательный вопрос.",
            ]
        }
        lst = responses.get(noise_type) or responses.get('meaningless')
        return random.choice(lst) if lst else "Пожалуйста, задайте конкретный вопрос."

    def _get_prompt_template(self) -> str:
        """Шаблон PromptTemplate для анализа вопросов."""
        return """Ты — эксперт по анализу вопросов для системы внутренней документации компании.

Проанализируй следующий вопрос пользователя:
Вопрос: "{question}"

Контекст (если есть): {context_str}

Выполни анализ по следующим критериям:
1. ЯСНОСТЬ: Насколько понятен вопрос?
2. СПЕЦИФИЧНОСТЬ: Достаточно ли конкретен вопрос?
3. ГРАММАТИКА: Есть ли грамматические ошибки?
4. ПОЛНОТА: Содержит ли вопрос всю необходимую информацию?
5. КОНТЕКСТ: Нужен ли дополнительный контекст?

Предоставь анализ в следующем JSON формате:
{{
  "clarity_score": число_0_1,
  "specificity_score": число_0_1,
  "grammar_score": число_0_1,
  "completeness_score": число_0_1,
  "context_score": число_0_1,
  "overall_score": число_0_1,
  "issues": [
    {{
      "type": "clarity|specificity|grammar|completeness|context|noise",
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
    
    @with_timeout(60.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        """Основная логика анализа вопроса."""
        question = context.original_query or ""
        # Шум-фильтрация
        is_noise, noise_type, suggested_response = self._is_noise_query(question)
        if is_noise:
            # Возвращаем структуру для шумового запроса
            result = {
                "is_noise_query": True,
                "noise_type": noise_type,
                "suggested_response": suggested_response,
                # Оценки по критериям для шумовых: обычно нулевые или нейтральные
                "clarity_score": 0.0,
                "specificity_score": 0.0,
                "grammar_score": 1.0,
                "completeness_score": 0.0,
                "context_score": 0.0,
                "overall_score": 0.0,
                "issues": [
                    {
                        "type": "noise",
                        "description": f"Неинформативный запрос категории '{noise_type}'",
                        "severity": "high",
                        "suggestion": suggested_response
                    }
                ],
                "improved_question": suggested_response,
                "clarifying_questions": [],
                "topic_hints": []
            }
            return result

        # Pattern-анализ (длина, неоднозначность, грамматика)
        pattern_analysis = self._analyze_patterns(question)

        # Подготовка контекста для LLM: можем передать topic, предыдущие ответы и т.п.
        context_str = self._build_context_str(context)

        # LLM-анализ
        llm_analysis = await self._llm_analysis(question, context_str)

        # Объединяем результаты
        result = self._combine_analysis(pattern_analysis, llm_analysis)

        # Явно отмечаем, что это содержательный запрос
        result["is_noise_query"] = False
        result["noise_type"] = None
        result["suggested_response"] = None
        result["should_process_further"] = True

        # Генерируем рекомендации на основе объединённого анализа
        result["recommendations"] = self._generate_recommendations(result, pattern_analysis)

        return result

    def _analyze_patterns(self, question: str) -> Dict[str, Any]:
        """Анализ вопроса с помощью паттернов и эвристик."""
        return {
            "length_analysis": self._analyze_length(question),
            "ambiguity_analysis": self._analyze_ambiguity(question),
            "grammar_analysis": self._analyze_grammar(question),
            "question_type": self._determine_question_type(question)
        }

    def _analyze_length(self, question: str) -> Dict[str, Any]:
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
        """Анализ неоднозначности по паттернам."""
        ambiguity_issues: List[str] = []
        for pattern in self.ambiguity_patterns:
            matches = re.findall(pattern, question.lower())
            if matches:
                ambiguity_issues.extend(matches)
        score = min(len(ambiguity_issues) * 0.2, 1.0)
        return {
            "ambiguous_terms": ambiguity_issues,
            "ambiguity_score": score,
            "has_ambiguity": bool(ambiguity_issues)
        }

    def _analyze_grammar(self, question: str) -> Dict[str, Any]:
        """Простейший анализ грамматики по паттернам."""
        grammar_issues: List[str] = []
        for pattern in self.grammar_patterns:
            if re.search(pattern, question):
                grammar_issues.append(pattern)
        starts_with_capital = bool(question and question[0].isupper())
        ends_with_question_mark = bool(question and question.strip().endswith("?"))
        # Оцениваем: уменьшаем за каждую проблему
        grammar_score = max(0.0, 1.0 - len(grammar_issues) * 0.2)
        return {
            "grammar_issues": grammar_issues,
            "starts_with_capital": starts_with_capital,
            "ends_with_question_mark": ends_with_question_mark,
            "grammar_score": grammar_score
        }

    def _determine_question_type(self, question: str) -> str:
        """Определение типа вопроса по ключевым словам."""
        ql = question.lower()
        if any(word in ql for word in ['что', 'какой', 'какая', 'какое']):
            return "definition"
        if any(word in ql for word in ['как', 'каким образом']):
            return "instruction"
        if any(word in ql for word in ['где', 'куда']):
            return "location"
        if any(word in ql for word in ['когда', 'во сколько']):
            return "time"
        if any(word in ql for word in ['почему', 'зачем']):
            return "reason"
        if any(word in ql for word in ['можно ли', 'возможно ли']):
            return "possibility"
        return "general"

    def _build_context_str(self, context: AgentContext) -> str:
        """
        Формирует строку контекста: topic, предыдущий ответ, enriched_context и др.
        """
        parts: List[str] = []
        if context.topic:
            parts.append(f"Тема: {context.topic}")
        # Если есть обогащённый контекст
        enriched = context.metadata.get("enriched_context")
        if isinstance(enriched, dict) and enriched:
            try:
                enriched_str = json.dumps(enriched, ensure_ascii=False)
            except:
                enriched_str = str(enriched)
            parts.append(f"Обогащённый контекст: {enriched_str}")
        # Если есть предыдущий ответ
        last_ans = context.metadata.get("answer_text") or context.metadata.get("final_answer")
        if isinstance(last_ans, str) and last_ans.strip():
            snippet = last_ans.strip().replace("\n", " ")
            snippet = snippet[:200]
            parts.append(f"Последний ответ: {snippet}")
        if parts:
            return " | ".join(parts)
        else:
            return "Нет дополнительного контекста."

    async def _llm_analysis(self, question: str, context_str: str) -> Dict[str, Any]:
        """
        Анализ вопроса с помощью LLM.
        """
        try:
            prompt = self.prompt_template.format(
                question=question,
                context_str=context_str or "нет"
            )
            # Вызываем через invoke_llm
            response = await self.invoke_llm(prompt)
        except Exception as e:
            self.logger.error(f"QuestionCriticAgent: ошибка LLM: {e}")
            return self._create_fallback_analysis(question)

        parsed = self.parse_json_response(response.strip())
        if not isinstance(parsed, dict):
            self.logger.warning("QuestionCriticAgent: LLM вернул не JSON-объект")
            return self._create_fallback_analysis(question)
        # Валидация структуры и типов полей
        return self._validate_llm_response(parsed, question)

    def _create_fallback_analysis(self, question: str) -> Dict[str, Any]:
        """Возвращает упрощённый анализ при сбое LLM."""
        # Простая эвристика: средние оценки
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
                    "description": "Анализ выполнен в упрощённом режиме из-за ошибки LLM",
                    "severity": "low",
                    "suggestion": "Попробуйте повторить позже"
                }
            ],
            "improved_question": question,
            "clarifying_questions": [],
            "topic_hints": []
        }

    def _validate_llm_response(self, parsed: Dict[str, Any], original_question: str) -> Dict[str, Any]:
        """
        Проверяем, что в parsed есть все ключи с корректными типами;
        при отсутствии или неверном типе подставляем дефолт.
        """
        default = self._create_fallback_analysis(original_question)
        out: Dict[str, Any] = {}

        # Оценки
        for key in ("clarity_score", "specificity_score", "grammar_score", "completeness_score", "context_score", "overall_score"):
            val = parsed.get(key)
            if isinstance(val, (int, float)):
                try:
                    fv = float(val)
                    out[key] = max(0.0, min(1.0, fv))
                except:
                    out[key] = default[key]
            else:
                out[key] = default[key]

        # issues: список dict с определёнными полями
        issues = parsed.get("issues")
        if isinstance(issues, list):
            valid_issues = []
            for item in issues:
                if not isinstance(item, dict):
                    continue
                type_ = item.get("type")
                desc = item.get("description")
                sev = item.get("severity")
                sugg = item.get("suggestion")
                if all(isinstance(f, str) for f in (type_, desc, sev, sugg)):
                    # severity: проверяем значение
                    if sev not in ("low", "medium", "high"):
                        sev = "low"
                    valid_issues.append({
                        "type": type_,
                        "description": desc,
                        "severity": sev,
                        "suggestion": sugg
                    })
            out["issues"] = valid_issues
        else:
            out["issues"] = default["issues"]

        # improved_question
        iq = parsed.get("improved_question")
        if isinstance(iq, str) and iq.strip():
            out["improved_question"] = iq.strip()
        else:
            out["improved_question"] = original_question

        # clarifying_questions
        cq = parsed.get("clarifying_questions")
        if isinstance(cq, list):
            out["clarifying_questions"] = [str(x) for x in cq if isinstance(x, str) and x.strip()]
        else:
            out["clarifying_questions"] = []

        # topic_hints
        th = parsed.get("topic_hints")
        if isinstance(th, list):
            out["topic_hints"] = [str(x) for x in th if isinstance(x, str) and x.strip()]
        else:
            out["topic_hints"] = []

        return out

    def _combine_analysis(self, pattern_analysis: Dict[str, Any], llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Объединяем результаты паттерн-анализа и LLM-анализа.
        Сохраняем pattern_analysis внутри результата, корректируем оценки при необходимости.
        """
        combined = {}
        # Копируем LLM-поля
        combined.update(llm_analysis)
        # Добавляем детали паттерн-анализа для отладки/логики
        combined["pattern_analysis"] = pattern_analysis

        # Пример корректировки: 
        # Если слишком короткий по паттерну, уменьшаем completeness
        length_info = pattern_analysis.get("length_analysis", {})
        if length_info.get("is_too_short") and combined.get("completeness_score", 1.0) > 0.5:
            combined["completeness_score"] = 0.5
        # Если неоднозначность, уменьшаем clarity
        ambiguity_info = pattern_analysis.get("ambiguity_analysis", {})
        if ambiguity_info.get("has_ambiguity") and combined.get("clarity_score", 1.0) > 0.6:
            combined["clarity_score"] = 0.6
        # Пересчёт overall_score как среднее
        scores = [
            combined.get("clarity_score", 0.0),
            combined.get("specificity_score", 0.0),
            combined.get("grammar_score", 0.0),
            combined.get("completeness_score", 0.0),
            combined.get("context_score", 0.0)
        ]
        combined["overall_score"] = sum(scores) / len(scores) if scores else 0.0

        return combined

    def _generate_recommendations(self, analysis: Dict[str, Any], pattern_analysis: Dict[str, Any]) -> List[str]:
        """
        Генерируем рекомендации по улучшению вопроса на основе оценок.
        """
        recommendations: List[str] = []
        if analysis.get("clarity_score", 1.0) < 0.7:
            recommendations.append("Сделайте вопрос более чётким и понятным")
        if analysis.get("specificity_score", 1.0) < 0.7:
            recommendations.append("Добавьте больше конкретных деталей")
        if analysis.get("completeness_score", 1.0) < 0.7:
            recommendations.append("Предоставьте больше контекста")
        if analysis.get("grammar_score", 1.0) < 0.7:
            recommendations.append("Проверьте грамматику и пунктуацию")
        # Паттерн-рекомендации
        length_info = pattern_analysis.get("length_analysis", {})
        if length_info.get("is_too_short"):
            recommendations.append("Расширьте вопрос — он слишком краток")
        if length_info.get("is_too_long"):
            recommendations.append("Сократите вопрос — он слишком длинный")
        ambiguity_info = pattern_analysis.get("ambiguity_analysis", {})
        if ambiguity_info.get("has_ambiguity"):
            recommendations.append("Избегайте неоднозначных формулировок")
        return recommendations

    @with_timeout(5.0)
    async def _postprocess(self, result_data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """
        Сохраняем результат анализа в metadata, чтобы orchestrator мог его использовать.
        Например: context.metadata['critic_analysis'] = result_data
        Также выставляем подсказки для retry, если overall_score низкий.
        """
        context.metadata["critic_analysis"] = result_data
        # Подсказки для orchestrator
        overall = result_data.get("overall_score", 0.0)
        if overall < 0.7:
            # Если низкая ясность или полнота, можно пытаться улучшить запрос
            if result_data.get("clarity_score", 1.0) < 0.7 or result_data.get("completeness_score", 1.0) < 0.7:
                context.metadata.setdefault("quality_hints", {})["needs_query_improvement"] = True
        return result_data

    def _calculate_confidence(self, result_data: Any, context: AgentContext) -> float:
        """Расчёт уверенности на основе overall_score или шумового флага."""
        if not isinstance(result_data, dict):
            return 0.0
        # Для шумовых запросов высокая уверенность в классификации как шум
        if result_data.get("is_noise_query", False):
            return 0.95
        overall = result_data.get("overall_score", 0.0)
        # Если есть pattern_analysis и issues, небольшое повышение
        has_pattern = "pattern_analysis" in result_data
        has_issues = isinstance(result_data.get("issues"), list)
        conf = overall
        if has_pattern and has_issues:
            conf = min(conf + 0.1, 1.0)
        return max(0.0, min(1.0, conf))


# Фабрика для создания агента
def create_question_critic_agent(config: Dict[str, Any] = None, langfuse_client=None) -> QuestionCriticAgent:
    """Фабричная функция для создания QuestionCriticAgent."""
    default_config = {
        "model_name": settings.LLM_MODEL_NAME,
        "temperature": 0.3,
        "ollama_base_url": str(settings.LLM_BASE_URL),
        # Можно добавить: "enable_pattern_analysis": True, "enable_noise_filtering": True и т.д.
    }
    if config:
        default_config.update(config)
    return QuestionCriticAgent(default_config, langfuse_client)
