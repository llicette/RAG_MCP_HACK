"""
Главный оркестратор для управления потоком агентов через LangGraph
"""
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
import asyncio
import logging
from datetime import datetime
import json

from ..agents.base_agent import AgentContext, AgentResult, BaseAgent, AgentManager

class WorkflowState(TypedDict):
    """Состояние workflow в LangGraph"""
    user_id: str
    session_id: str
    original_query: str
    processed_query: Optional[str]
    topic: Optional[str]
    documents: Optional[List[Dict]]
    final_answer: Optional[str]
    confidence_scores: Dict[str, float]
    agent_results: Dict[str, Any]
    metadata: Dict[str, Any]
    current_step: str
    max_iterations: int
    iteration_count: int
    errors: List[str]

class DocumentationOrchestrator:
    """Главный оркестратор для AI Documentation Assistant"""
    
    def __init__(self, 
                 agent_manager: AgentManager,
                 config: Dict[str, Any]):
        self.agent_manager = agent_manager
        self.config = config
        self.logger = logging.getLogger("orchestrator")
        self.graph = None
        self._build_workflow()
    
    def _build_workflow(self):
        """Построение LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Добавляем узлы (агенты)
        workflow.add_node("question_critic", self._question_critic_node)
        workflow.add_node("query_rewriter", self._query_rewriter_node)
        workflow.add_node("topic_classifier", self._topic_classifier_node)
        workflow.add_node("document_retriever", self._document_retriever_node)
        workflow.add_node("answer_generator", self._answer_generator_node)
        workflow.add_node("quality_checker", self._quality_checker_node)
        workflow.add_node("context_enricher", self._context_enricher_node)
        
        # Устанавливаем точку входа
        workflow.set_entry_point("question_critic")
        
        # Определяем переходы
        workflow.add_edge("question_critic", "query_rewriter")
        workflow.add_edge("query_rewriter", "topic_classifier")
        workflow.add_edge("topic_classifier", "document_retriever")
        workflow.add_edge("document_retriever", "context_enricher")
        workflow.add_edge("context_enricher", "answer_generator")
        workflow.add_edge("answer_generator", "quality_checker")
        
        # Условный переход после проверки качества
        workflow.add_conditional_edges(
            "quality_checker",
            self._should_retry,
            {
                "retry": "query_rewriter",  # Повторить с улучшенным запросом
                "finish": END,             # Завершить
                "enhance": "context_enricher"  # Улучшить контекст
            }
        )
        
        # Компилируем граф
        self.graph = workflow.compile()
        
        self.logger.info("Workflow построен успешно")
    
    async def process_query(self, 
                          user_id: str,
                          session_id: str,
                          query: str,
                          max_iterations: int = 3) -> Dict[str, Any]:
        """Основной метод обработки запроса"""
        
        initial_state = WorkflowState(
            user_id=user_id,
            session_id=session_id,
            original_query=query,
            processed_query=None,
            topic=None,
            documents=None,
            final_answer=None,
            confidence_scores={},
            agent_results={},
            metadata={
                "start_time": datetime.now().isoformat(),
                "workflow_version": "1.0"
            },
            current_step="question_critic",
            max_iterations=max_iterations,
            iteration_count=0,
            errors=[]
        )
        
        try:
            self.logger.info(f"Начинаю обработку запроса: {query}")
            
            # Выполняем workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            # Формируем результат
            result = {
                "success": True,
                "answer": final_state.get("final_answer"),
                "confidence": self._calculate_overall_confidence(final_state),
                "metadata": {
                    **final_state.get("metadata", {}),
                    "end_time": datetime.now().isoformat(),
                    "total_iterations": final_state.get("iteration_count", 0),
                    "confidence_scores": final_state.get("confidence_scores", {}),
                    "errors": final_state.get("errors", [])
                }
            }
            
            self.logger.info(f"Запрос обработан успешно за {final_state.get('iteration_count', 0)} итераций")
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "end_time": datetime.now().isoformat(),
                    "error_details": str(e)
                }
            }
    
    async def _question_critic_node(self, state: WorkflowState) -> WorkflowState:
        """Узел критика вопросов"""
        context = self._state_to_context(state)
        result = await self.agent_manager.execute_agent("question_critic", context)
        
        state["agent_results"]["question_critic"] = result.data
        state["confidence_scores"]["question_critic"] = result.confidence
        state["current_step"] = "question_critic"
        
        if not result.success:
            state["errors"].append(f"Question Critic: {result.error_message}")
        
        return state
    
    async def _query_rewriter_node(self, state: WorkflowState) -> WorkflowState:
        """Узел переписывания запросов"""
        context = self._state_to_context(state)
        result = await self.agent_manager.execute_agent("query_rewriter", context)
        
        if result.success and result.data:
            state["processed_query"] = result.data.get("rewritten_query", state["original_query"])
        
        state["agent_results"]["query_rewriter"] = result.data
        state["confidence_scores"]["query_rewriter"] = result.confidence
        state["current_step"] = "query_rewriter"
        
        if not result.success:
            state["errors"].append(f"Query Rewriter: {result.error_message}")
        
        return state
    
    async def _topic_classifier_node(self, state: WorkflowState) -> WorkflowState:
        """Узел классификации тем"""
        context = self._state_to_context(state)
        result = await self.agent_manager.execute_agent("topic_classifier", context)
        
        if result.success and result.data:
            state["topic"] = result.data.get("topic")
        
        state["agent_results"]["topic_classifier"] = result.data
        state["confidence_scores"]["topic_classifier"] = result.confidence
        state["current_step"] = "topic_classifier"
        
        if not result.success:
            state["errors"].append(f"Topic Classifier: {result.error_message}")
        
        return state
    
    async def _document_retriever_node(self, state: WorkflowState) -> WorkflowState:
        """Узел поиска документов"""
        context = self._state_to_context(state)
        result = await self.agent_manager.execute_agent("document_retriever", context)
        
        if result.success and result.data:
            state["documents"] = result.data.get("documents", [])
        
        state["agent_results"]["document_retriever"] = result.data
        state["confidence_scores"]["document_retriever"] = result.confidence
        state["current_step"] = "document_retriever"
        
        if not result.success:
            state["errors"].append(f"Document Retriever: {result.error_message}")
        
        return state
    
    async def _context_enricher_node(self, state: WorkflowState) -> WorkflowState:
        """Узел обогащения контекста"""
        context = self._state_to_context(state)
        result = await self.agent_manager.execute_agent("context_enricher", context)
        
        if result.success and result.data:
            # Обновляем документы с дополнительным контекстом
            enriched_docs = result.data.get("enriched_documents", state.get("documents", []))
            state["documents"] = enriched_docs
        
        state["agent_results"]["context_enricher"] = result.data
        state["confidence_scores"]["context_enricher"] = result.confidence
        state["current_step"] = "context_enricher"
        
        if not result.success:
            state["errors"].append(f"Context Enricher: {result.error_message}")
        
        return state
    
    async def _answer_generator_node(self, state: WorkflowState) -> WorkflowState:
        """Узел генерации ответов"""
        context = self._state_to_context(state)
        result = await self.agent_manager.execute_agent("answer_generator", context)
        
        if result.success and result.data:
            state["final_answer"] = result.data.get("answer")
        
        state["agent_results"]["answer_generator"] = result.data
        state["confidence_scores"]["answer_generator"] = result.confidence
        state["current_step"] = "answer_generator"
        
        if not result.success:
            state["errors"].append(f"Answer Generator: {result.error_message}")
        
        return state
    
    async def _quality_checker_node(self, state: WorkflowState) -> WorkflowState:
        """Узел проверки качества"""
        context = self._state_to_context(state)
        result = await self.agent_manager.execute_agent("quality_checker", context)
        
        state["agent_results"]["quality_checker"] = result.data
        state["confidence_scores"]["quality_checker"] = result.confidence
        state["current_step"] = "quality_checker"
        state["iteration_count"] += 1
        
        if not result.success:
            state["errors"].append(f"Quality Checker: {result.error_message}")
        
        return state
    
    def _should_retry(self, state: WorkflowState) -> str:
        """Определяет, нужно ли повторить обработку"""
        quality_result = state.get("agent_results", {}).get("quality_checker", {})
        quality_score = state.get("confidence_scores", {}).get("quality_checker", 0.0)
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        
        # Если превышено максимальное количество итераций
        if iteration_count >= max_iterations:
            self.logger.info(f"Достигнуто максимальное количество итераций: {max_iterations}")
            return "finish"
        
        # Если качество низкое и есть рекомендации по улучшению
        if quality_score < 0.7:
            if quality_result and quality_result.get("needs_context_enrichment"):
                self.logger.info("Требуется обогащение контекста")
                return "enhance"
            elif quality_result and quality_result.get("needs_query_improvement"):
                self.logger.info("Требуется улучшение запроса")
                return "retry"
        
        # Если качество приемлемое
        if quality_score >= 0.7:
            self.logger.info(f"Качество приемлемое: {quality_score}")
            return "finish"
        
        # По умолчанию повторяем
        self.logger.info("Повторяем обработку")
        return "retry"
    
    def _state_to_context(self, state: WorkflowState) -> AgentContext:
        """Преобразование состояния workflow в контекст агента"""
        return AgentContext(
            user_id=state["user_id"],
            session_id=state["session_id"],
            original_query=state["original_query"],
            processed_query=state.get("processed_query"),
            topic=state.get("topic"),
            documents=state.get("documents"),
            metadata={
                **state.get("metadata", {}),
                "current_step": state.get("current_step"),
                "iteration_count": state.get("iteration_count", 0),
                "agent_results": state.get("agent_results", {}),
                "confidence_scores": state.get("confidence_scores", {})
            }
        )
    
    def _calculate_overall_confidence(self, state: WorkflowState) -> float:
        """Расчет общей уверенности в результате"""
        scores = state.get("confidence_scores", {})
        if not scores:
            return 0.0
        
        # Взвешенная средняя с учетом важности агентов
        weights = {
            "question_critic": 0.1,
            "query_rewriter": 0.1,
            "topic_classifier": 0.15,
            "document_retriever": 0.25,
            "context_enricher": 0.15,
            "answer_generator": 0.35,
            "quality_checker": 0.3
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for agent, score in scores.items():
            weight = weights.get(agent, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    async def get_workflow_status(self, session_id: str) -> Dict[str, Any]:
        """Получение статуса выполнения workflow"""
        # Здесь можно добавить логику для отслеживания активных workflow
        return {
            "session_id": session_id,
            "status": "completed",  # или "running", "failed" и т.д.
            "timestamp": datetime.now().isoformat()
        }
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Получение метрик workflow"""
        return {
            "total_processed": 0,  # Можно добавить счетчики
            "average_confidence": 0.0,
            "average_processing_time": 0.0,
            "success_rate": 0.0
        }

class WorkflowBuilder:
    """Строитель для создания различных workflow конфигураций"""
    
    @staticmethod
    def create_fast_workflow(agent_manager: AgentManager) -> DocumentationOrchestrator:
        """Создание быстрого workflow для простых запросов"""
        config = {
            "workflow_type": "fast",
            "max_iterations": 1,
            "skip_quality_check": False,
            "parallel_processing": True
        }
        return DocumentationOrchestrator(agent_manager, config)
    
    @staticmethod
    def create_thorough_workflow(agent_manager: AgentManager) -> DocumentationOrchestrator:
        """Создание тщательного workflow для сложных запросов"""
        config = {
            "workflow_type": "thorough",
            "max_iterations": 5,
            "skip_quality_check": False,
            "parallel_processing": False,
            "enable_context_enrichment": True
        }
        return DocumentationOrchestrator(agent_manager, config)
    
    @staticmethod
    def create_debug_workflow(agent_manager: AgentManager) -> DocumentationOrchestrator:
        """Создание workflow для отладки"""
        config = {
            "workflow_type": "debug",
            "max_iterations": 1,
            "skip_quality_check": True,
            "verbose_logging": True,
            "save_intermediate_results": True
        }
        return DocumentationOrchestrator(agent_manager, config)