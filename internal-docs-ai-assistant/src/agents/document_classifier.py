import re
import os
from typing import Dict, Any, List, Tuple, Optional, Union
from enum import Enum
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import magic
import hashlib
from collections import Counter, defaultdict

# ML и NLP библиотеки
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# LangChain
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# PDF и документы
import PyPDF2
import docx
from bs4 import BeautifulSoup
import pandas as pd

from agents.base_agent import BaseAgent, AgentContext, AgentResult, with_retry, with_timeout


class DocumentType(Enum):
    """Типы документов"""
    POLICY = "policy"  # Политики и положения
    PROCEDURE = "procedure"  # Процедуры и инструкции
    REGULATION = "regulation"  # Регламенты
    FORM = "form"  # Формы и бланки
    MANUAL = "manual"  # Руководства
    FAQ = "faq"  # Часто задаваемые вопросы
    ANNOUNCEMENT = "announcement"  # Объявления и новости
    REPORT = "report"  # Отчеты
    CONTRACT = "contract"  # Договоры
    GUIDE = "guide"  # Справочники и гиды
    TRAINING = "training"  # Обучающие материалы
    TECHNICAL = "technical"  # Техническая документация
    OTHER = "other"  # Прочие документы

class DocumentPriority(Enum):
    """Приоритет документа"""
    CRITICAL = 1  # Критически важные
    HIGH = 2     # Высокий приоритет
    MEDIUM = 3   # Средний приоритет
    LOW = 4      # Низкий приоритет

class DocumentAudience(Enum):
    """Целевая аудитория"""
    ALL_EMPLOYEES = "all_employees"  # Все сотрудники
    MANAGEMENT = "management"        # Руководство
    HR = "hr"                       # HR отдел
    IT = "it"                       # IT отдел
    FINANCE = "finance"             # Финансовый отдел
    LEGAL = "legal"                 # Юридический отдел
    SPECIFIC_ROLE = "specific_role" # Определенная роль
    NEW_EMPLOYEES = "new_employees" # Новые сотрудники

class DocumentClassifierAgent(BaseAgent):
    """Агент для классификации документов внутренней документации"""
    
    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("document_classifier", config, langfuse_client)
        
        # LLM для анализа содержимого
        self.llm = Ollama(
            model=self.get_config("model_name", "llama3.1:8b"),
            temperature=self.get_config("temperature", 0.1),
            base_url=self.get_config("ollama_base_url", "http://localhost:11434")
        )
        
        # Настройка классификационных правил
        self.classification_rules = self._setup_classification_rules()
        
        # Ключевые слова для классификации
        self.type_keywords = self._setup_type_keywords()
        self.priority_keywords = self._setup_priority_keywords()
        self.audience_keywords = self._setup_audience_keywords()
        
        # TF-IDF для семантического анализа
        self.tfidf_vectorizer = None
        self.document_type_vectors = None
        self._initialize_vectorizer()
        
        # Стоп-слова
        try:
            self.stop_words = set(stopwords.words('russian'))
        except:
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('russian'))
        
        # Сплиттер текста
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Промпты для LLM анализа
        self.classification_prompt = PromptTemplate(
            input_variables=["document_text", "filename", "metadata"],
            template=self._get_classification_prompt()
        )
        
        self.content_analysis_prompt = PromptTemplate(
            input_variables=["document_content", "document_type"],
            template=self._get_content_analysis_prompt()
        )
        
        # Кэш для классификации
        self.classification_cache = {}
        
    def _setup_classification_rules(self) -> Dict[str, Any]:
        """Настройка правил классификации документов"""
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
                    "Q:", "A:", "FAQ"
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
        """Ключевые слова для определения типа документа"""
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
        """Ключевые слова для определения приоритета"""
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
        """Ключевые слова для определения целевой аудитории"""
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
        """Инициализация TF-IDF векторизатора для типов документов"""
        try:
            # Создаем корпус из ключевых слов каждого типа
            type_texts = []
            type_names = []
            
            for doc_type, keywords in self.type_keywords.items():
                type_text = " ".join(keywords)
                type_texts.append(type_text)
                type_names.append(doc_type)
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words=list(self.stop_words),
                ngram_range=(1, 2),
                min_df=1
            )
            
            self.document_type_vectors = self.tfidf_vectorizer.fit_transform(type_texts)
            self.type_names = type_names
            
        except Exception as e:
            self.logger.warning(f"Не удалось инициализировать TF-IDF: {e}")
            self.tfidf_vectorizer = None
    
    def _get_classification_prompt(self) -> str:
        """Промпт для классификации документа"""
        # Экранируем фигурные скобки JSON-примера двойными {{ и }}
        return (
            "Ты - эксперт по классификации внутренних документов компании.\n"
            "\n"
            "Имя файла: {filename}\n"
            "Метаданные: {metadata}\n"
            "\n"
            "Содержимое документа (первые 2000 символов):\n{document_text}\n"
            "\n"
            "Проанализируй документ и определи:\n"
            "\n"
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
            "\n"
            "2. Приоритет документа:\n"
            "   - critical: Критически важный\n"
            "   - high: Высокий приоритет\n"
            "   - medium: Средний приоритет\n"
            "   - low: Низкий приоритет\n"
            "\n"
            "3. Целевая аудитория:\n"
            "   - all_employees: Все сотрудники\n"
            "   - management: Руководство\n"
            "   - hr: HR отдел\n"
            "   - it: IT отдел\n"
            "   - finance: Финансовый отдел\n"
            "   - legal: Юридический отдел\n"
            "   - specific_role: Определенная роль\n"
            "   - new_employees: Новые сотрудники\n"
            "\n"
            "4. Дополнительные характеристики:\n"
            "   - Требует ли документ регулярного обновления\n"
            "   - Содержит ли конфиденциальную информацию\n"
            "   - Есть ли срок действия\n"
            "\n"
            "Ответь в JSON формате:\n"
            "{{\n"
            "  \"document_type\": \"тип_документа\",\n"
            "  \"priority\": \"приоритет\",\n"
            "  \"target_audience\": [\"аудитория1\", \"аудитория2\"],\n"
            "  \"confidence\": число_0_1,\n"
            "  \"requires_updates\": true/false,\n"
            "  \"confidential\": true/false,\n"
            "  \"has_expiry\": true/false,\n"
            "  \"estimated_validity_months\": число_или_null,\n"
            "  \"key_topics\": [\"тема1\", \"тема2\", \"тема3\"],\n"
            "  \"document_purpose\": \"краткое_описание_назначения\",\n"
            "  \"reasoning\": \"объяснение_классификации\"\n"
            "}}"
        )
    
    def _get_content_analysis_prompt(self) -> str:
        """Промпт для анализа содержимого документа"""
        # Экранируем JSON-пример двойными {{ }}
        return (
            "Проанализируй содержимое документа типа \"{document_type}\" и извлеки ключевую информацию.\n"
            "\n"
            "Содержимое документа:\n{document_content}\n"
            "\n"
            "Определи:\n"
            "1. Основные разделы и структуру документа\n"
            "2. Ключевые термины и понятия\n"
            "3. Действия, которые описаны в документе\n"
            "4. Связанные процессы или документы\n"
            "5. Важные даты, сроки, числовые показатели\n"
            "\n"
            "Ответь в JSON формате:\n"
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
    
    @with_timeout(60.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> Dict[str, Any]:
        """Основная логика классификации документа"""
        
        # Получение информации о документе
        document_info = context.metadata.get("document_info", {})
        if not document_info:
            raise ValueError("Отсутствует информация о документе в контексте")
        
        # Извлечение текста документа
        document_text = await self._extract_text(document_info)
        if not document_text:
            raise ValueError("Не удалось извлечь текст из документа")
        
        # Создание хэша для кэширования
        doc_hash = self._create_document_hash(document_info, document_text[:1000])
        
        # Проверка кэша
        if doc_hash in self.classification_cache:
            self.logger.info("Результат найден в кэше")
            return self.classification_cache[doc_hash]
        
        # Многоуровневая классификация
        filename_analysis = self._analyze_filename(document_info)
        content_analysis = self._analyze_content(document_text)
        structure_analysis = self._analyze_structure(document_text)
        metadata_analysis = self._analyze_metadata(document_info)
        
        # Приоритет и аудитория
        priority_analysis = self._analyze_priority(document_text)
        audience_analysis = self._analyze_audience(document_text)
        
        # LLM анализ
        llm_analysis = await self._llm_classification(
            document_text, document_info, context
        )
        
        # Семантический анализ
        semantic_analysis = self._semantic_analysis(document_text)
        
        # Объединение результатов
        final_classification = self._combine_analyses(
            filename_analysis, content_analysis, structure_analysis,
            metadata_analysis, llm_analysis, semantic_analysis,
            priority_analysis, audience_analysis
        )
        
        # Детальный анализ содержимого
        if final_classification.get("confidence", 0) > 0.5:
            content_details = await self._detailed_content_analysis(
                document_text, final_classification.get("document_type")
            )
            final_classification["content_analysis"] = content_details
        
        # Сохранение в кэш
        self.classification_cache[doc_hash] = final_classification
        
        return final_classification
    
    async def _extract_text(self, document_info: Dict[str, Any]) -> str:
        """Извлечение текста из документа"""
        try:
            file_path = document_info.get("file_path")
            file_type = document_info.get("file_type", "").lower()
            
            if not file_path or not os.path.exists(file_path):
                # Если передан сам текст
                return document_info.get("content", "")
            
            # Определение типа файла
            if not file_type:
                file_type = magic.from_file(file_path, mime=True)
            
            if file_type == "application/pdf" or file_path.endswith('.pdf'):
                return await self._extract_pdf_text(file_path)
            elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"] or file_path.endswith('.docx'):
                return await self._extract_docx_text(file_path)
            elif file_type == "text/html" or file_path.endswith('.html'):
                return await self._extract_html_text(file_path)
            elif file_type == "text/plain" or file_path.endswith('.txt'):
                return await self._extract_txt_text(file_path)
            else:
                self.logger.warning(f"Неподдерживаемый тип файла: {file_type}")
                return ""
                
        except Exception as e:
            self.logger.error(f"Ошибка извлечения текста: {e}")
            return ""
    
    async def _extract_pdf_text(self, file_path: str) -> str:
        """Извлечение текста из PDF"""
        def extract():
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text() or ''
                    text += page_text + "\n"
                return text
        
        return await asyncio.to_thread(extract)
    
    async def _extract_docx_text(self, file_path: str) -> str:
        """Извлечение текста из DOCX"""
        def extract():
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        return await asyncio.to_thread(extract)
    
    async def _extract_html_text(self, file_path: str) -> str:
        """Извлечение текста из HTML"""
        def extract():
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                return soup.get_text()
        
        return await asyncio.to_thread(extract)
    
    async def _extract_txt_text(self, file_path: str) -> str:
        """Извлечение текста из TXT"""
        def extract():
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        
        return await asyncio.to_thread(extract)
    
    def _create_document_hash(self, document_info: Dict, text_sample: str) -> str:
        """Создание хэша документа для кэширования"""
        hash_string = f"{document_info.get('filename', '')}{document_info.get('size', '')}{text_sample}"
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def _analyze_filename(self, document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ имени файла"""
        filename = document_info.get("filename", "").lower()
        
        type_scores = {}
        
        # Проверка паттернов имен файлов
        for doc_type, patterns in self.classification_rules["filename_patterns"].items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, filename):
                    score += 2
            type_scores[doc_type] = score
        
        # Проверка ключевых слов в имени
        for doc_type, keywords in self.type_keywords.items():
            score = type_scores.get(doc_type, 0)
            for keyword in keywords:
                if keyword in filename:
                    score += 1
            type_scores[doc_type] = score
        
        if not type_scores or max(type_scores.values()) == 0:
            return {"method": "filename", "document_type": DocumentType.OTHER, "confidence": 0.0}
        
        best_type = max(type_scores.items(), key=lambda x: x[1])[0]
        confidence = min(type_scores[best_type] / 5, 1.0)
        
        return {
            "method": "filename",
            "document_type": best_type,
            "confidence": confidence,
            "type_scores": type_scores
        }
    
    def _analyze_content(self, document_text: str) -> Dict[str, Any]:
        """Анализ содержимого документа"""
        text_lower = document_text.lower()
        
        type_scores = {}
        
        # Проверка индикаторов содержимого
        for doc_type, indicators in self.classification_rules.get("content_indicators", {}).items():
            score = 0
            for indicator in indicators:
                if indicator in text_lower:
                    score += 2
            type_scores[doc_type] = score
        
        # Проверка ключевых слов
        for doc_type, keywords in self.type_keywords.items():
            score = type_scores.get(doc_type, 0)
            for keyword in keywords:
                count = text_lower.count(keyword)
                score += count
            type_scores[doc_type] = score
        
        if not type_scores or max(type_scores.values()) == 0:
            return {"method": "content", "document_type": DocumentType.OTHER, "confidence": 0.0}
        
        best_type = max(type_scores.items(), key=lambda x: x[1])[0]
        total_score = sum(type_scores.values())
        confidence = type_scores[best_type] / total_score if total_score > 0 else 0.0
        
        return {
            "method": "content",
            "document_type": best_type,
            "confidence": min(confidence, 1.0),
            "type_scores": type_scores
        }
    
    def _analyze_structure(self, document_text: str) -> Dict[str, Any]:
        """Анализ структуры документа"""
        text_lower = document_text.lower()
        
        # Поиск структурных паттернов
        structure_scores = {}
        
        for doc_type, patterns in self.classification_rules.get("structural_patterns", {}).items():
            score = 0
            for pattern in patterns:
                if pattern in text_lower:
                    score += 1
            structure_scores[doc_type] = score
        
        # Анализ структурных элементов
        structural_elements = {
            "has_numbered_sections": bool(re.search(r'\d+\.\s+[А-Яа-я]', document_text)),
            "has_bullet_points": bool(re.search(r'[•\-\*]\s+', document_text)),
            "has_tables": bool(re.search(r'\|.*\|', document_text)),
            "has_signatures": bool(re.search(r'подпись|signature', text_lower)),
            "has_dates": bool(re.search(r'\d{1,2}\.\d{1,2}\.\d{4}', document_text)),
            "has_forms": bool(re.search(r'_____+|\.\.\.\.\.+', document_text))
        }
        
        # Дополнительная оценка на основе структуры
        if structural_elements["has_forms"]:
            structure_scores[DocumentType.FORM] = structure_scores.get(DocumentType.FORM, 0) + 3
        
        if structural_elements["has_numbered_sections"]:
            structure_scores[DocumentType.PROCEDURE] = structure_scores.get(DocumentType.PROCEDURE, 0) + 2
            structure_scores[DocumentType.MANUAL] = structure_scores.get(DocumentType.MANUAL, 0) + 2
        
        if not structure_scores:
            return {"method": "structure", "document_type": DocumentType.OTHER, "confidence": 0.0}
        
        best_type = max(structure_scores.items(), key=lambda x: x[1])[0]
        confidence = min(structure_scores[best_type] / 5, 1.0)
        
        return {
            "method": "structure",
            "document_type": best_type,
            "confidence": confidence,
            "structure_scores": structure_scores,
            "structural_elements": structural_elements
        }
    
    def _analyze_metadata(self, document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ метаданных документа"""
        # Пример метаданных: filename, size, created_at, modified_at, author, tags
        metadata = {}
        if 'filename' in document_info:
            metadata['filename'] = document_info['filename']
        if 'size' in document_info:
            metadata['size'] = document_info['size']
        # Даты создания/изменения
        for date_field in ['created_at', 'modified_at', 'creation_date', 'modified_date']:
            if date_field in document_info:
                metadata[date_field] = document_info[date_field]
        if 'author' in document_info:
            metadata['author'] = document_info['author']
        # Здесь можно добавлять дополнительные правила по метаданным
        # Если есть теги, метки конфиденциальности
        confidential = False
        if 'confidential' in document_info:
            confidential = bool(document_info.get('confidential'))
        metadata['confidential_flag'] = confidential
        
        # По умолчанию не определяем тип из метаданных
        return {"method": "metadata", "metadata": metadata, "confidential_flag": confidential, "document_type": None, "confidence": 0.0}
    
    def _analyze_priority(self, document_text: str) -> Dict[str, Any]:
        """Анализ приоритета документа на основе ключевых слов"""
        text_lower = document_text.lower()
        scores = {}
        for prio, keywords in self.priority_keywords.items():
            score = 0
            for kw in keywords:
                score += text_lower.count(kw)
            scores[prio] = score
        if not scores or max(scores.values()) == 0:
            return {"priority": DocumentPriority.MEDIUM, "confidence": 0.0, "priority_scores": scores}
        best = max(scores.items(), key=lambda x: x[1])[0]
        total = sum(scores.values())
        confidence = scores[best] / total if total > 0 else 0.0
        return {"priority": best, "confidence": min(confidence, 1.0), "priority_scores": scores}
    
    def _analyze_audience(self, document_text: str) -> Dict[str, Any]:
        """Анализ целевой аудитории документа"""
        text_lower = document_text.lower()
        audience_counts = {}
        for aud, keywords in self.audience_keywords.items():
            count = 0
            for kw in keywords:
                count += text_lower.count(kw)
            if count > 0:
                audience_counts[aud] = count
        # Сортируем по убыванию
        sorted_aud = sorted(audience_counts.items(), key=lambda x: x[1], reverse=True)
        audiences = [aud for aud, cnt in sorted_aud]
        return {"target_audience": audiences, "audience_counts": audience_counts}
    
    async def _llm_classification(self, document_text: str, document_info: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Классификация с помощью LLM"""
        try:
            sample = document_text[:2000]
            metadata_str = str(document_info)
            prompt = self.classification_prompt.format(
                document_text=sample,
                filename=document_info.get('filename', ''),
                metadata=metadata_str
            )
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            import json
            try:
                result = json.loads(response)
                # Преобразуем строковые типы в Enum, если нужно
                if 'document_type' in result:
                    result['document_type'] = DocumentType(result['document_type']) if result['document_type'] in DocumentType._value2member_map_ else DocumentType.OTHER
                if 'priority' in result:
                    result['priority'] = DocumentPriority[result['priority'].upper()] if result['priority'].upper() in DocumentPriority.__members__ else None
                if 'target_audience' in result:
                    # Сохраняем список строк, перевести в Enum при необходимости
                    pass
                return result
            except json.JSONDecodeError as e:
                self.logger.warning(f"Не удалось распарсить JSON ответ от LLM: {e}")
                return {}
        except Exception as e:
            self.logger.error(f"Ошибка при LLM классификации: {e}")
            return {}
    
    def _semantic_analysis(self, document_text: str) -> Dict[str, Any]:
        """Семантическая классификация на основе TF-IDF"""
        if not self.tfidf_vectorizer or not self.document_type_vectors:
            return {"method": "semantic", "document_type": DocumentType.OTHER, "confidence": 0.0}
        # Преобразуем текст: берем первые N символов или ключевые предложения
        try:
            # Разбиваем на предложения и берем первые несколько
            sents = sent_tokenize(document_text)
            sample = ' '.join(sents[:10])
            vec = self.tfidf_vectorizer.transform([sample])
            sims = cosine_similarity(vec, self.document_type_vectors)[0]
            # Найти лучший индекс
            best_idx = int(np.argmax(sims))
            best_type = self.type_names[best_idx]
            confidence = float(sims[best_idx])
            # Создаем словарь типов к значениям
            sim_scores = {self.type_names[i]: float(sims[i]) for i in range(len(sims))}
            return {"method": "semantic", "document_type": best_type, "confidence": confidence, "similarity_scores": sim_scores}
        except Exception as e:
            self.logger.warning(f"Ошибка семантического анализа: {e}")
            return {"method": "semantic", "document_type": DocumentType.OTHER, "confidence": 0.0}
    
    def _combine_analyses(self, filename_analysis: Dict[str, Any], content_analysis: Dict[str, Any],
                          structure_analysis: Dict[str, Any], metadata_analysis: Dict[str, Any],
                          llm_analysis: Dict[str, Any], semantic_analysis: Dict[str, Any],
                          priority_analysis: Dict[str, Any], audience_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Объединение всех анализов в финальную классификацию"""
        # Собираем голосование за тип документа
        scores = defaultdict(float)
        analyses = [filename_analysis, content_analysis, structure_analysis, semantic_analysis, llm_analysis]
        weights = {
            'filename': 0.5,
            'content': 1.0,
            'structure': 0.8,
            'semantic': 1.0,
            'llm': 1.5
        }
        for a in analyses:
            dtype = a.get('document_type')
            if isinstance(dtype, DocumentType):
                method = a.get('method', '')
                weight = weights.get(method, 1.0)
                scores[dtype] += a.get('confidence', 0.0) * weight
        # Выбор лучшего типа
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])[0]
            # нормируем confidence: отношение к сумме всех
            total = sum(scores.values())
            confidence = scores[best_type] / total if total > 0 else 0.0
        else:
            best_type = DocumentType.OTHER
            confidence = 0.0
        # Приоритет и аудитория
        priority = priority_analysis.get('priority')
        priority_conf = priority_analysis.get('confidence')
        audiences = audience_analysis.get('target_audience', [])
        # Дополнительные характеристики
        text_lower = ''
        requires_updates = False
        confidential = metadata_analysis.get('confidential_flag', False)
        has_expiry = False
        estimated_months = None
        key_topics = []
        document_purpose = ''
        reasoning_parts = []
        reasoning_parts.append(f"Выбран тип {best_type.value} с уверенностью {confidence:.2f}")
        # Проверка на обновление
        result = {
            'document_type': best_type,
            'priority': priority,
            'target_audience': audiences,
            'confidence': confidence,
            'requires_updates': requires_updates,
            'confidential': confidential,
            'has_expiry': has_expiry,
            'estimated_validity_months': estimated_months,
            'key_topics': key_topics,
            'document_purpose': document_purpose,
            'reasoning': '; '.join(reasoning_parts)
        }
        return result
    
    async def _detailed_content_analysis(self, document_text: str, document_type: DocumentType) -> Dict[str, Any]:
        """Детальный анализ содержимого с LLM"""
        # Разбиваем текст на чанки
        docs = [Document(page_content=chunk) for chunk in self.text_splitter.split_text(document_text)]
        main_sections = set()
        key_terms = set()
        described_actions = set()
        related_processes = set()
        important_dates = set()
        numerical_data = set()
        structure_scores = []
        readability_scores = []
        completeness_scores = []
        # Ограничим количество чанков для анализа
        for doc in docs[:5]:
            prompt = self.content_analysis_prompt.format(
                document_content=doc.page_content,
                document_type=document_type.value
            )
            try:
                response = await asyncio.to_thread(self.llm.invoke, prompt)
                import json
                res = json.loads(response)
                for k in ['main_sections', 'key_terms', 'described_actions', 'related_processes', 'important_dates', 'numerical_data']:
                    items = res.get(k, [])
                    if isinstance(items, list):
                        if k == 'main_sections': main_sections.update(items)
                        elif k == 'key_terms': key_terms.update(items)
                        elif k == 'described_actions': described_actions.update(items)
                        elif k == 'related_processes': related_processes.update(items)
                        elif k == 'important_dates': important_dates.update(items)
                        elif k == 'numerical_data': numerical_data.update(items)
                # Считаем метрики
                if 'document_structure_quality' in res:
                    structure_scores.append(res.get('document_structure_quality', 0.0))
                if 'readability_score' in res:
                    readability_scores.append(res.get('readability_score', 0.0))
                if 'completeness_score' in res:
                    completeness_scores.append(res.get('completeness_score', 0.0))
            except Exception:
                continue
        # Усреднение метрик
        def avg(lst): return sum(lst)/len(lst) if lst else None
        return {
            'main_sections': list(main_sections),
            'key_terms': list(key_terms),
            'described_actions': list(described_actions),
            'related_processes': list(related_processes),
            'important_dates': list(important_dates),
            'numerical_data': list(numerical_data),
            'document_structure_quality': avg(structure_scores) if avg(structure_scores) is not None else 0.0,
            'readability_score': avg(readability_scores) if avg(readability_scores) is not None else 0.0,
            'completeness_score': avg(completeness_scores) if avg(completeness_scores) is not None else 0.0
        }

# Фабрика для создания агента

def create_document_classifier_agent(config: Dict[str, Any] = None, 
                                     langfuse_client=None) -> DocumentClassifierAgent:
    default_config = {
        "model_name": "llama3.1:8b",
        "temperature": 0.1,
        "ollama_base_url": "http://localhost:11434",
        "max_analysis_time": 30.0
    }
    if config:
        default_config.update(config)
    return DocumentClassifierAgent(default_config, langfuse_client)
