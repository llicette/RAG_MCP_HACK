import re
import json
import uuid
import asyncio
import os
import hashlib
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime
from collections import defaultdict

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.llms import Ollama  # если у вас старая версия, можно: from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import PyPDF2
import docx
from bs4 import BeautifulSoup

from agents.base_agent import BaseAgent, AgentContext, AgentResult, with_retry, with_timeout
from configs.settings import settings


class DocumentType(Enum):
    POLICY = "policy"
    PROCEDURE = "procedure"
    REGULATION = "regulation"
    FORM = "form"
    MANUAL = "manual"
    FAQ = "faq"
    ANNOUNCEMENT = "announcement"
    REPORT = "report"
    CONTRACT = "contract"
    GUIDE = "guide"
    TRAINING = "training"
    TECHNICAL = "technical"
    OTHER = "other"


class DocumentPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DocumentAudience(Enum):
    ALL_EMPLOYEES = "all_employees"
    MANAGEMENT = "management"
    HR = "hr"
    IT = "it"
    FINANCE = "finance"
    LEGAL = "legal"
    SPECIFIC_ROLE = "specific_role"
    NEW_EMPLOYEES = "new_employees"


class DocumentClassifierAgent(BaseAgent):
    """Агент для классификации документов внутренней документации."""

    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("document_classifier", config, langfuse_client)

        # Инициализация LLM
        model_name = self.get_config("model_name", settings.LLM_MODEL_NAME)
        base_url = self.get_config("ollama_base_url", None) or str(settings.LLM_BASE_URL)
        temperature = float(self.get_config("temperature", 0.1))
        self.llm = Ollama(model=model_name, temperature=temperature, base_url=base_url)

        # Настройка правил и ключевых слов
        self.classification_rules = self._setup_classification_rules()
        self.type_keywords = self._setup_type_keywords()
        self.priority_keywords = self._setup_priority_keywords()
        self.audience_keywords = self._setup_audience_keywords()

        # NLP: стоп-слова, nltk
        try:
            self.stop_words = set(stopwords.words('russian'))
        except Exception:
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('russian'))

        # TF-IDF векторизатор для семантики типов документов
        self.tfidf_vectorizer = None
        self.document_type_vectors = None
        self.type_names = []
        self._initialize_vectorizer()

        # Текстовый сплиттер для детального анализа
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(self.get_config("chunk_size", 1000)),
            chunk_overlap=int(self.get_config("chunk_overlap", 200)),
            length_function=len
        )

        # PromptTemplates с экранированием двойными {{ }}
        self.classification_prompt = PromptTemplate(
            input_variables=["document_text", "filename", "metadata_str"],
            template=self._get_classification_prompt()
        )
        self.content_analysis_prompt = PromptTemplate(
            input_variables=["document_content", "document_type"],
            template=self._get_content_analysis_prompt()
        )

        # Внутренний кэш: ключ — хэш документа, значение — результат классификации
        self.classification_cache: Dict[str, Dict[str, Any]] = {}

    def _setup_classification_rules(self) -> Dict[str, Any]:
        """Настройка правил классификации документов."""
        return {
            "filename_patterns": {
                DocumentType.POLICY: [
                    r"политика.*\.pdf", r"policy.*\.pdf", r"положение.*\.pdf",
                    r"устав.*\.pdf", r"кодекс.*\.pdf"
                ],
                DocumentType.PROCEDURE: [
                    r"процедура.*\.pdf", r"инструкция.*\.pdf", r"алгоритм.*\.pdf",
                    r"порядок.*\.pdf", r"схема.*\.pdf"
                ],
                DocumentType.REGULATION: [
                    r"регламент.*\.pdf", r"стандарт.*\.pdf", r"норматив.*\.pdf",
                    r"требования.*\.pdf"
                ],
                DocumentType.FORM: [
                    r"форма.*\.pdf", r"бланк.*\.pdf", r"заявление.*\.pdf",
                    r"анкета.*\.pdf", r"template.*\.pdf"
                ],
                DocumentType.MANUAL: [
                    r"руководство.*\.pdf", r"manual.*\.pdf", r"справочник.*\.pdf",
                    r"пособие.*\.pdf"
                ],
                DocumentType.FAQ: [
                    r"faq.*\.pdf", r"вопросы.*\.pdf", r"ответы.*\.pdf",
                    r"q&a.*\.pdf"
                ],
                DocumentType.ANNOUNCEMENT: [
                    r"объявление.*\.pdf", r"новости.*\.pdf", r"уведомление.*\.pdf",
                    r"информация.*\.pdf"
                ],
                DocumentType.REPORT: [
                    r"отчет.*\.pdf", r"report.*\.pdf", r"анализ.*\.pdf",
                    r"статистика.*\.pdf"
                ],
                DocumentType.CONTRACT: [
                    r"договор.*\.pdf", r"соглашение.*\.pdf", r"контракт.*\.pdf",
                    r"contract.*\.pdf"
                ]
            },
            "content_indicators": {
                DocumentType.POLICY: [
                    "настоящая политика", "данная политика", "политика компании",
                    "принципы работы", "корпоративные ценности"
                ],
                DocumentType.PROCEDURE: [
                    "выполните следующие шаги", "пошаговая инструкция",
                    "алгоритм действий", "последовательность операций"
                ],
                DocumentType.REGULATION: [
                    "настоящий регламент", "требования к", "стандарты",
                    "нормативные требования"
                ],
                DocumentType.FORM: [
                    "заполните поля", "данная форма", "бланк заявления",
                    "подпись", "дата заполнения"
                ],
                DocumentType.FAQ: [
                    "часто задаваемые вопросы", "вопрос:", "ответ:",
                    "q&a", "faq", "ответы на вопросы"
                ]
            },
            "structural_patterns": {
                DocumentType.POLICY: ["введение", "область применения", "ответственность"],
                DocumentType.PROCEDURE: ["цель", "область применения", "описание процедуры"],
                DocumentType.REGULATION: ["общие положения", "требования", "контроль"],
                DocumentType.MANUAL: ["содержание", "глава", "раздел"],
                DocumentType.REPORT: ["аннотация", "выводы", "рекомендации"]
            }
        }

    def _setup_type_keywords(self) -> Dict[DocumentType, List[str]]:
        """Ключевые слова для типов документов."""
        return {
            DocumentType.POLICY: [
                "политика", "положение", "принципы", "ценности", "этика",
                "стандарты поведения", "кодекс", "устав", "миссия"
            ],
            DocumentType.PROCEDURE: [
                "процедура", "инструкция", "алгоритм", "порядок действий",
                "пошаговый", "последовательность", "этапы", "методика"
            ],
            DocumentType.REGULATION: [
                "регламент", "стандарт", "норматив", "требования",
                "спецификация", "критерии", "правила", "нормы"
            ],
            DocumentType.FORM: [
                "форма", "бланк", "заявление", "анкета", "шаблон",
                "образец", "заполнение", "подпись", "дата"
            ],
            DocumentType.MANUAL: [
                "руководство", "справочник", "пособие", "гид",
                "инструкция пользователя", "описание", "manual"
            ],
            DocumentType.FAQ: [
                "faq", "часто задаваемые вопросы", "вопросы и ответы",
                "q&a", "популярные вопросы", "ответы на вопросы"
            ],
            DocumentType.ANNOUNCEMENT: [
                "объявление", "уведомление", "новости", "информация",
                "сообщение", "извещение", "анонс"
            ],
            DocumentType.REPORT: [
                "отчет", "анализ", "исследование", "статистика",
                "результаты", "данные", "показатели", "метрики"
            ],
            DocumentType.CONTRACT: [
                "договор", "соглашение", "контракт", "условия",
                "обязательства", "права", "ответственность"
            ],
            DocumentType.GUIDE: [
                "гид", "путеводитель", "инструкция", "советы",
                "рекомендации", "как", "пошагово"
            ],
            DocumentType.TRAINING: [
                "обучение", "тренинг", "курс", "урок", "занятие",
                "материалы", "презентация", "лекция"
            ],
            DocumentType.TECHNICAL: [
                "техническая документация", "спецификация", "api",
                "интеграция", "настройка", "конфигурация"
            ]
        }

    def _setup_priority_keywords(self) -> Dict[DocumentPriority, List[str]]:
        """Ключевые слова для приоритета."""
        return {
            DocumentPriority.CRITICAL: [
                "критически важно", "обязательно", "немедленно",
                "срочно", "безопасность", "конфиденциально",
                "экстренно", "чрезвычайно", "vital", "critical"
            ],
            DocumentPriority.HIGH: [
                "важно", "приоритет", "первоочередно", "высокий",
                "существенно", "значимо", "ключевой", "основной"
            ],
            DocumentPriority.MEDIUM: [
                "рекомендуется", "желательно", "стандартный",
                "обычный", "регулярный", "плановый"
            ],
            DocumentPriority.LOW: [
                "дополнительно", "опционально", "при необходимости",
                "справочно", "информационно", "вспомогательный"
            ]
        }

    def _setup_audience_keywords(self) -> Dict[DocumentAudience, List[str]]:
        """Ключевые слова для аудитории."""
        return {
            DocumentAudience.ALL_EMPLOYEES: [
                "все сотрудники", "весь персонал", "каждый сотрудник",
                "общие правила", "корпоративные", "для всех"
            ],
            DocumentAudience.MANAGEMENT: [
                "руководство", "менеджмент", "директор", "начальник",
                "управление", "руководители", "топ-менеджмент"
            ],
            DocumentAudience.HR: [
                "hr", "кадры", "персонал", "человеческие ресурсы",
                "найм", "увольнение", "аттестация", "обучение"
            ],
            DocumentAudience.IT: [
                "it", "информационные технологии", "системные",
                "программисты", "разработчики", "техподдержка"
            ],
            DocumentAudience.FINANCE: [
                "финансы", "бухгалтерия", "экономика", "бюджет",
                "расходы", "доходы", "отчетность", "налоги"
            ],
            DocumentAudience.LEGAL: [
                "юридический", "правовой", "юрист", "договоры",
                "соглашения", "правовые", "legal"
            ],
            DocumentAudience.NEW_EMPLOYEES: [
                "новые сотрудники", "новички", "стажеры",
                "адаптация", "введение", "первый день"
            ]
        }

    def _initialize_vectorizer(self):
        """Инициализация TF-IDF для семантики."""
        try:
            # Минимальный корпус: ключевые слова каждого типа
            type_texts = []
            for doc_type, keywords in self.type_keywords.items():
                type_texts.append(" ".join(keywords))
                self.type_names.append(doc_type)
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words=list(self.stop_words),
                ngram_range=(1, 2),
                min_df=1
            )
            self.document_type_vectors = self.tfidf_vectorizer.fit_transform(type_texts)
        except Exception as e:
            self.logger.warning(f"Не удалось инициализировать TF-IDF: {e}")
            self.tfidf_vectorizer = None
            self.document_type_vectors = None

    def _get_classification_prompt(self) -> str:
        """Prompt для LLM-классификации документа."""
        # Экранируем двойными {{ }} JSON-образец
        return (
            "Ты — эксперт по классификации внутренних документов компании.\n"
            "Имя файла: {filename}\n"
            "Метаданные: {metadata_str}\n"
            "Содержимое документа (первые 2000 символов):\n{document_text}\n"
            "Проанализируй документ и определи:\n"
            "1. Тип документа из списка:\n"
            "   - policy: Политики и положения компании\n"
            "   - procedure: Процедуры и инструкции\n"
            "   - regulation: Регламенты и стандарты\n"
            "   - form: Формы и бланки\n"
            "   - manual: Руководства и справочники\n"
            "   - faq: Часто задаваемые вопросы\n"
            "   - announcement: Объявления и новости\n"
            "   - report: Отчеты и аналитика\n"
            "   - contract: Договоры и соглашения\n"
            "   - guide: Гиды и путеводители\n"
            "   - training: Обучающие материалы\n"
            "   - technical: Техническая документация\n"
            "   - other: Прочие документы\n"
            "2. Приоритет документа:\n"
            "   - critical: Критически важный\n"
            "   - high: Высокий приоритет\n"
            "   - medium: Средний приоритет\n"
            "   - low: Низкий приоритет\n"
            "3. Целевая аудитория:\n"
            "   - all_employees: Все сотрудники\n"
            "   - management: Руководство\n"
            "   - hr: HR отдел\n"
            "   - it: IT отдел\n"
            "   - finance: Финансовый отдел\n"
            "   - legal: Юридический отдел\n"
            "   - specific_role: Определенная роль\n"
            "   - new_employees: Новые сотрудники\n"
            "4. Дополнительные характеристики:\n"
            "   - requires_updates (true/false)\n"
            "   - confidential (true/false)\n"
            "   - has_expiry (true/false)\n"
            "   - estimated_validity_months (число или null)\n"
            "5. Ключевые темы (key_topics) — список важных тем из документа.\n"
            "6. Назначение документа (document_purpose) — короткое описание.\n"
            "7. Reasoning — объяснение выбора.\n"
            "Ответь строго в JSON формате:\n"
            "{{\n"
            "  \"document_type\": \"тип_документа\",\n"
            "  \"priority\": \"приоритет\",\n"
            "  \"target_audience\": [\"аудитория1\", \"аудитория2\"],\n"
            "  \"confidence\": число_0_1,\n"
            "  \"requires_updates\": true/false,\n"
            "  \"confidential\": true/false,\n"
            "  \"has_expiry\": true/false,\n"
            "  \"estimated_validity_months\": число_или_null,\n"
            "  \"key_topics\": [\"тема1\", \"тема2\"],\n"
            "  \"document_purpose\": \"краткое описание\",\n"
            "  \"reasoning\": \"объяснение\"\n"
            "}}"
        )

    def _get_content_analysis_prompt(self) -> str:
        """Prompt для детального анализа содержимого документа."""
        return (
            "Проанализируй содержимое документа типа \"{document_type}\" и извлеки ключевую информацию.\n"
            "Содержимое документа:\n{document_content}\n"
            "Определи:\n"
            "1. Основные разделы и структуру документа\n"
            "2. Ключевые термины и понятия\n"
            "3. Действия, которые описаны в документе\n"
            "4. Связанные процессы или документы\n"
            "5. Важные даты, сроки, числовые показатели\n"
            "Ответь строго в JSON формате:\n"
            "{{\n"
            "  \"main_sections\": [\"раздел1\", \"раздел2\"],\n"
            "  \"key_terms\": [\"термин1\", \"термин2\"],\n"
            "  \"described_actions\": [\"действие1\", \"действие2\"],\n"
            "  \"related_processes\": [\"процесс1\", \"процесс2\"],\n"
            "  \"important_dates\": [\"дата1\", \"дата2\"],\n"
            "  \"numerical_data\": [\"показатель1\", \"показатель2\"],\n"
            "  \"document_structure_quality\": число_0_1,\n"
            "  \"readability_score\": число_0_1,\n"
            "  \"completeness_score\": число_0_1\n"
            "}}"
        )

    @with_timeout(120.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        """
        Основная логика классификации документа.
        Ожидается, что в context.metadata["document_info"] есть словарь с ключами:
        - file_path (опционально) или content (строка)
        - filename, size, author, и т.п.
        """
        # Определяем уникальный идентификатор документа: либо из metadata, либо генерируем
        document_info = context.metadata.get("document_info")
        if not isinstance(document_info, dict):
            raise ValueError("DocumentClassifierAgent: в context.metadata нет ключа 'document_info' или он неверный")

        # Генерируем или берём doc_id для кэша
        doc_id = context.metadata.get("doc_id") or str(uuid.uuid4())
        # Создадим хэш по filename+size+фрагменту текста
        # Сначала пытаемся извлечь текст
        document_text = await self._extract_text(document_info)
        if not document_text:
            raise ValueError("DocumentClassifierAgent: не удалось извлечь текст")

        text_sample = document_text[:2000]
        hash_input = f"{document_info.get('filename','')}{document_info.get('size','')}{text_sample}"
        doc_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()

        # Кэширование в памяти
        if doc_hash in self.classification_cache:
            self.logger.info("DocumentClassifierAgent: результат найден в кэше")
            return self.classification_cache[doc_hash]

        # Анализы
        filename_analysis = self._analyze_filename(document_info)
        content_analysis = self._analyze_content(document_text)
        structure_analysis = self._analyze_structure(document_text)
        metadata_analysis = self._analyze_metadata(document_info)
        priority_analysis = self._analyze_priority(document_text)
        audience_analysis = self._analyze_audience(document_text)
        llm_analysis = await self._llm_classification(document_text, document_info, context)
        semantic_analysis = self._semantic_analysis(document_text)

        # Объединяем
        final_classification = self._combine_analyses(
            filename_analysis, content_analysis, structure_analysis,
            metadata_analysis, llm_analysis, semantic_analysis,
            priority_analysis, audience_analysis
        )

        # Если уверенность достаточна, проводим детальный анализ содержимого
        try:
            conf = float(final_classification.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        if conf > 0.5:
            details = await self._detailed_content_analysis(document_text, final_classification.get("document_type"))
            final_classification["content_analysis"] = details

        # Сохраняем в кэш
        self.classification_cache[doc_hash] = final_classification

        return final_classification

    async def _extract_text(self, document_info: Dict[str, Any]) -> str:
        """Извлекает текст из файла или берёт из document_info['content']."""
        file_path = document_info.get("file_path")
        content_override = document_info.get("content")
        if content_override and isinstance(content_override, str) and not file_path:
            return content_override

        if not file_path or not os.path.exists(file_path):
            # Если поле content не задано или файл не найден
            return content_override or ""

        # По расширению или mime-типу
        lower = file_path.lower()
        try:
            if lower.endswith(".pdf"):
                return await self._extract_pdf_text(file_path)
            elif lower.endswith(".docx"):
                return await self._extract_docx_text(file_path)
            elif lower.endswith(".html") or lower.endswith(".htm"):
                return await self._extract_html_text(file_path)
            elif lower.endswith(".txt"):
                return await self._extract_txt_text(file_path)
            else:
                # Можно расширить для других типов
                self.logger.warning(f"DocumentClassifierAgent: неподдерживаемый тип файла {file_path}")
                return ""
        except Exception as e:
            self.logger.error(f"DocumentClassifierAgent: ошибка при извлечении текста: {e}")
            return ""

    async def _extract_pdf_text(self, file_path: str) -> str:
        def extract():
            text = ""
            try:
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        text += page_text + "\n"
            except Exception as e:
                # Логируем, но возвращаем то, что удалось
                self.logger.error(f"DocumentClassifierAgent: PDF extract error: {e}")
            return text
        return await asyncio.to_thread(extract)

    async def _extract_docx_text(self, file_path: str) -> str:
        def extract():
            text = ""
            try:
                doc = docx.Document(file_path)
                for p in doc.paragraphs:
                    text += p.text + "\n"
            except Exception as e:
                self.logger.error(f"DocumentClassifierAgent: DOCX extract error: {e}")
            return text
        return await asyncio.to_thread(extract)

    async def _extract_html_text(self, file_path: str) -> str:
        def extract():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html.parser")
                    return soup.get_text(separator="\n")
            except Exception as e:
                self.logger.error(f"DocumentClassifierAgent: HTML extract error: {e}")
                return ""
        return await asyncio.to_thread(extract)

    async def _extract_txt_text(self, file_path: str) -> str:
        def extract():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                self.logger.error(f"DocumentClassifierAgent: TXT extract error: {e}")
                return ""
        return await asyncio.to_thread(extract)

    def _analyze_filename(self, document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ по имени файла."""
        filename = (document_info.get("filename") or "").lower()
        type_scores = {}
        # По шаблонам
        for doc_type, patterns in self.classification_rules.get("filename_patterns", {}).items():
            score = 0
            for pat in patterns:
                if re.search(pat, filename):
                    score += 2
            type_scores[doc_type] = score
        # По ключевым словам
        for doc_type, keywords in self.type_keywords.items():
            sc = type_scores.get(doc_type, 0)
            for kw in keywords:
                if kw in filename:
                    sc += 1
            type_scores[doc_type] = sc
        if not type_scores or max(type_scores.values()) == 0:
            return {"method": "filename", "document_type": DocumentType.OTHER, "confidence": 0.0}
        best, best_score = max(type_scores.items(), key=lambda x: x[1])
        confidence = min(best_score / 5.0, 1.0)
        return {"method": "filename", "document_type": best, "confidence": confidence, "type_scores": type_scores}

    def _analyze_content(self, document_text: str) -> Dict[str, Any]:
        """Анализ содержимого по ключевым индикаторам и словам."""
        text_lower = document_text.lower()
        type_scores = {}
        # Индикаторы
        for doc_type, indicators in self.classification_rules.get("content_indicators", {}).items():
            sc = 0
            for ind in indicators:
                if ind in text_lower:
                    sc += 2
            type_scores[doc_type] = sc
        # По ключевым словам
        for doc_type, keywords in self.type_keywords.items():
            sc = type_scores.get(doc_type, 0)
            for kw in keywords:
                sc += text_lower.count(kw)
            type_scores[doc_type] = sc
        if not type_scores or max(type_scores.values()) == 0:
            return {"method": "content", "document_type": DocumentType.OTHER, "confidence": 0.0}
        best, best_score = max(type_scores.items(), key=lambda x: x[1])
        total = sum(type_scores.values())
        confidence = best_score / total if total > 0 else 0.0
        return {"method": "content", "document_type": best, "confidence": min(confidence, 1.0), "type_scores": type_scores}

    def _analyze_structure(self, document_text: str) -> Dict[str, Any]:
        """Анализ структуры документа."""
        text_lower = document_text.lower()
        structure_scores = {}
        # Паттерны
        for doc_type, patterns in self.classification_rules.get("structural_patterns", {}).items():
            sc = 0
            for pat in patterns:
                if pat in text_lower:
                    sc += 1
            structure_scores[doc_type] = sc
        # Элементы
        structural_elements = {
            "has_numbered_sections": bool(re.search(r'\d+\.\s+[А-ЯA-Za-z]', document_text)),
            "has_bullet_points": bool(re.search(r'[•\-\*]\s+', document_text)),
            "has_tables": bool(re.search(r'\|.*\|', document_text)),
            "has_signatures": bool(re.search(r'подпись|signature', text_lower)),
            "has_dates": bool(re.search(r'\d{1,2}\.\d{1,2}\.\d{4}', document_text)),
            "has_forms": bool(re.search(r'_{3,}', document_text))
        }
        # Уточнения
        if structural_elements.get("has_forms"):
            structure_scores[DocumentType.FORM] = structure_scores.get(DocumentType.FORM, 0) + 3
        if structural_elements.get("has_numbered_sections"):
            structure_scores[DocumentType.PROCEDURE] = structure_scores.get(DocumentType.PROCEDURE, 0) + 2
            structure_scores[DocumentType.MANUAL] = structure_scores.get(DocumentType.MANUAL, 0) + 2
        if not structure_scores:
            return {"method": "structure", "document_type": DocumentType.OTHER, "confidence": 0.0}
        best, best_score = max(structure_scores.items(), key=lambda x: x[1])
        confidence = min(best_score / 5.0, 1.0)
        return {
            "method": "structure",
            "document_type": best,
            "confidence": confidence,
            "structure_scores": structure_scores,
            "structural_elements": structural_elements
        }

    def _analyze_metadata(self, document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ метаданных документа."""
        # Собираем простые поля
        metadata = {}
        if 'filename' in document_info:
            metadata['filename'] = document_info['filename']
        if 'size' in document_info:
            metadata['size'] = document_info['size']
        for date_field in ['created_at', 'modified_at', 'creation_date', 'modified_date']:
            if date_field in document_info:
                metadata[date_field] = document_info[date_field]
        if 'author' in document_info:
            metadata['author'] = document_info['author']
        confidential = bool(document_info.get('confidential', False))
        metadata['confidential_flag'] = confidential
        # По умолчанию document_type не определяем здесь
        return {"method": "metadata", "metadata": metadata, "confidential_flag": confidential,
                "document_type": None, "confidence": 0.0}

    def _analyze_priority(self, document_text: str) -> Dict[str, Any]:
        """Анализ приоритета по ключевым словам."""
        text_lower = document_text.lower()
        scores = {}
        for prio, keywords in self.priority_keywords.items():
            sc = 0
            for kw in keywords:
                sc += text_lower.count(kw)
            scores[prio] = sc
        if not scores or max(scores.values()) == 0:
            return {"priority": DocumentPriority.MEDIUM.value, "confidence": 0.0, "priority_scores": scores}
        best, best_score = max(scores.items(), key=lambda x: x[1])
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.0
        return {"priority": best.value, "confidence": min(confidence, 1.0), "priority_scores": scores}

    def _analyze_audience(self, document_text: str) -> Dict[str, Any]:
        """Анализ целевой аудитории по ключевым словам."""
        text_lower = document_text.lower()
        audience_counts = {}
        for aud, keywords in self.audience_keywords.items():
            cnt = 0
            for kw in keywords:
                cnt += text_lower.count(kw)
            if cnt > 0:
                audience_counts[aud.value] = cnt
        # Сортировка
        sorted_aud = sorted(audience_counts.items(), key=lambda x: x[1], reverse=True)
        audiences = [aud for aud, _ in sorted_aud]
        return {"target_audience": audiences, "audience_counts": audience_counts}

    async def _llm_classification(self, document_text: str, document_info: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Классификация через LLM."""
        try:
            sample = document_text[:2000]
            metadata_str = json.dumps(document_info, ensure_ascii=False)
            prompt = self.classification_prompt.format(
                document_text=sample,
                filename=document_info.get('filename', ''),
                metadata_str=metadata_str
            )
            response = await self.invoke_llm(prompt)
            parsed = self.parse_json_response(response)
            if not isinstance(parsed, dict):
                return {}
            # Приведение типов: document_type и priority из строк в значения Enum.value
            out: Dict[str, Any] = {}
            # document_type
            dt = parsed.get("document_type")
            if isinstance(dt, str) and dt in DocumentType._value2member_map_:
                out["document_type"] = DocumentType(dt).value
            else:
                out["document_type"] = DocumentType.OTHER.value
            # priority
            pr = parsed.get("priority")
            if isinstance(pr, str) and pr.lower() in DocumentPriority._value2member_map_:
                out["priority"] = pr.lower()
            else:
                out["priority"] = None
            # target_audience
            ta = parsed.get("target_audience")
            if isinstance(ta, list):
                # оставляем только допустимые строки
                filtered = [str(x) for x in ta if isinstance(x, str)]
                out["target_audience"] = filtered
            # confidence
            try:
                conf = float(parsed.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            out["confidence"] = max(0.0, min(conf, 1.0))
            # requires_updates, confidential, has_expiry, estimated_validity_months
            out["requires_updates"] = bool(parsed.get("requires_updates", False))
            out["confidential"] = bool(parsed.get("confidential", False))
            out["has_expiry"] = bool(parsed.get("has_expiry", False))
            ev = parsed.get("estimated_validity_months")
            out["estimated_validity_months"] = ev if isinstance(ev, (int, float)) else None
            # key_topics
            kt = parsed.get("key_topics")
            if isinstance(kt, list):
                out["key_topics"] = [str(x) for x in kt]
            else:
                out["key_topics"] = []
            # document_purpose
            dp = parsed.get("document_purpose")
            out["document_purpose"] = str(dp) if dp is not None else ""
            # reasoning
            rn = parsed.get("reasoning")
            out["reasoning"] = str(rn) if rn is not None else ""
            return out
        except Exception as e:
            self.logger.error(f"DocumentClassifierAgent: ошибка LLM классификации: {e}")
            return {}

    def _semantic_analysis(self, document_text: str) -> Dict[str, Any]:
        """Семантическая классификация через TF-IDF."""
        if not self.tfidf_vectorizer or self.document_type_vectors is None:
            return {"method": "semantic", "document_type": DocumentType.OTHER.value, "confidence": 0.0}
        try:
            # Берём первые предложения
            sents = sent_tokenize(document_text)
            sample = " ".join(sents[:10]) if sents else document_text[:500]
            vec = self.tfidf_vectorizer.transform([sample])
            sims = cosine_similarity(vec, self.document_type_vectors)[0]
            idx = int(np.argmax(sims))
            best_type = self.type_names[idx]
            confidence = float(sims[idx])
            # similarity_scores
            sim_scores = {
                self.type_names[i].value: float(sims[i]) for i in range(len(sims))
            }
            return {"method": "semantic", "document_type": best_type.value, "confidence": min(confidence, 1.0), "similarity_scores": sim_scores}
        except Exception as e:
            self.logger.warning(f"DocumentClassifierAgent: ошибка семантического анализа: {e}")
            return {"method": "semantic", "document_type": DocumentType.OTHER.value, "confidence": 0.0}

    def _combine_analyses(self,
                          filename_analysis: Dict[str, Any],
                          content_analysis: Dict[str, Any],
                          structure_analysis: Dict[str, Any],
                          metadata_analysis: Dict[str, Any],
                          llm_analysis: Dict[str, Any],
                          semantic_analysis: Dict[str, Any],
                          priority_analysis: Dict[str, Any],
                          audience_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Объединяем все анализы в итоговую классификацию."""
        # Голосование за тип документа
        scores = defaultdict(float)
        analyses = [
            filename_analysis, content_analysis, structure_analysis,
            semantic_analysis, llm_analysis
        ]
        # Веса можно настраивать
        weights = {
            "filename": 0.5,
            "content": 1.0,
            "structure": 0.8,
            "semantic": 1.0,
            "llm": 1.5
        }
        for a in analyses:
            dt = a.get("document_type")
            method = a.get("method", "")
            if isinstance(dt, str):
                weight = weights.get(method, 1.0)
                try:
                    conf = float(a.get("confidence", 0.0))
                except:
                    conf = 0.0
                scores[dt] += conf * weight
        if scores:
            best_type, best_score = max(scores.items(), key=lambda x: x[1])
            total = sum(scores.values())
            confidence = best_score / total if total > 0 else 0.0
        else:
            best_type = DocumentType.OTHER.value
            confidence = 0.0

        # Приоритет и аудитория
        priority = priority_analysis.get("priority", DocumentPriority.MEDIUM.value)
        priority_conf = priority_analysis.get("confidence", 0.0)
        audiences = audience_analysis.get("target_audience", [])

        # Дополнительные характеристики: здесь можно расширить логику
        requires_updates = False
        confidential = metadata_analysis.get("confidential_flag", False)
        has_expiry = False
        estimated_months = None
        key_topics: List[str] = []
        document_purpose = ""
        reasoning = f"Выбран тип {best_type} с уверенностью {confidence:.2f}"

        result: Dict[str, Any] = {
            "document_type": best_type,
            "priority": priority,
            "target_audience": audiences,
            "confidence": min(max(confidence, 0.0), 1.0),
            "requires_updates": requires_updates,
            "confidential": confidential,
            "has_expiry": has_expiry,
            "estimated_validity_months": estimated_months,
            "key_topics": key_topics,
            "document_purpose": document_purpose,
            "reasoning": reasoning
        }
        return result

    async def _detailed_content_analysis(self, document_text: str, document_type: str) -> Dict[str, Any]:
        """Детальный анализ с LLM: разбиваем на чанки и агрегируем результаты."""
        # Разбиваем на чанки
        try:
            chunks = self.text_splitter.split_text(document_text)
        except Exception:
            chunks = [document_text]
        main_sections = set()
        key_terms = set()
        described_actions = set()
        related_processes = set()
        important_dates = set()
        numerical_data = set()
        structure_scores = []
        readability_scores = []
        completeness_scores = []

        # Ограничиваем число чанков для скорости
        for chunk in chunks[:5]:
            prompt = self.content_analysis_prompt.format(
                document_content=chunk,
                document_type=document_type
            )
            try:
                response = await self.invoke_llm(prompt)
                parsed = self.parse_json_response(response)
                if not isinstance(parsed, dict):
                    continue
                # Сбор полей
                for k, target_set in [
                    ("main_sections", main_sections),
                    ("key_terms", key_terms),
                    ("described_actions", described_actions),
                    ("related_processes", related_processes),
                    ("important_dates", important_dates),
                    ("numerical_data", numerical_data)
                ]:
                    items = parsed.get(k)
                    if isinstance(items, list):
                        for it in items:
                            try:
                                target_set.add(str(it))
                            except:
                                pass
                # Метрики
                dsq = parsed.get("document_structure_quality")
                if isinstance(dsq, (int, float)):
                    structure_scores.append(float(dsq))
                rs = parsed.get("readability_score")
                if isinstance(rs, (int, float)):
                    readability_scores.append(float(rs))
                cs = parsed.get("completeness_score")
                if isinstance(cs, (int, float)):
                    completeness_scores.append(float(cs))
            except Exception as e:
                self.logger.warning(f"DocumentClassifierAgent: детальный анализ не удался для чанка: {e}")
                continue

        def avg(lst: List[float]) -> Optional[float]:
            return sum(lst)/len(lst) if lst else None

        return {
            "main_sections": list(main_sections),
            "key_terms": list(key_terms),
            "described_actions": list(described_actions),
            "related_processes": list(related_processes),
            "important_dates": list(important_dates),
            "numerical_data": list(numerical_data),
            "document_structure_quality": avg(structure_scores) if avg(structure_scores) is not None else 0.0,
            "readability_score": avg(readability_scores) if avg(readability_scores) is not None else 0.0,
            "completeness_score": avg(completeness_scores) if avg(completeness_scores) is not None else 0.0
        }

    async def _postprocess(self, result_data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        # Можно сохранить результат классификации в metadata, если нужно
        context.metadata["last_document_classification"] = result_data
        return result_data

    def _calculate_confidence(self, result_data: Any, context: AgentContext) -> float:
        # Возвращаем итоговое confidence, если есть
        if isinstance(result_data, dict):
            try:
                conf = float(result_data.get("confidence", 0.0))
                return min(max(conf, 0.0), 1.0)
            except:
                return 0.0
        return 0.0


def create_document_classifier_agent(config: Dict[str, Any] = None, langfuse_client=None) -> DocumentClassifierAgent:
    default_config = {
        "model_name": settings.LLM_MODEL_NAME,
        "temperature": 0.1,
        "ollama_base_url": str(settings.LLM_BASE_URL),
        # Можно добавить другие default-параметры (chunk sizes и т.п.)
    }
    if config:
        default_config.update(config)
    return DocumentClassifierAgent(default_config, langfuse_client)
