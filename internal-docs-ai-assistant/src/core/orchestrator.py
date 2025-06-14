from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END

import asyncio
import logging
from datetime import datetime

from agents.base_agent import AgentContext, AgentManager

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
        self.config = config or {}
        self.logger = logging.getLogger("orchestrator")
        self.logger.setLevel(logging.DEBUG)
        
        self.graph = None
        self._build_workflow()
    
    def _build_workflow(self):
        workflow = StateGraph(WorkflowState)
        # Добавляем узлы
        workflow.add_node("question_critic", self._question_critic_node)
        workflow.add_node("query_rewriter", self._query_rewriter_node)
        workflow.add_node("topic_classifier", self._topic_classifier_node)
        workflow.add_node("document_retriever", self._document_retriever_node)
        
        # Контекстное обогащение
        if self.config.get("enable_context_enrichment", True):
            workflow.add_node("context_enricher", self._context_enricher_node)
            after_retriever = "context_enricher"
        else:
            after_retriever = None
        
        workflow.add_node("answer_generator", self._answer_generator_node)
        
        skip_qc = self.config.get("skip_quality_check", False)
        if not skip_qc:
            workflow.add_node("quality_checker", self._quality_checker_node)
        
        # Точка входа
        workflow.set_entry_point("question_critic")
        # Последовательные переходы
        workflow.add_edge("question_critic", "query_rewriter")
        workflow.add_edge("query_rewriter", "topic_classifier")
        workflow.add_edge("topic_classifier", "document_retriever")
        if after_retriever:
            workflow.add_edge("document_retriever", "context_enricher")
            workflow.add_edge("context_enricher", "answer_generator")
        else:
            workflow.add_edge("document_retriever", "answer_generator")
        
        if skip_qc:
            # После генератора сразу завершение
            workflow.add_edge("answer_generator", END)
        else:
            workflow.add_edge("answer_generator", "quality_checker")
            # Условный переход после проверки качества
            workflow.add_conditional_edges(
                "quality_checker",
                self._should_retry,
                {
                    "retry": "query_rewriter",      
                    "finish": END,                  
                    "enhance": after_retriever or "query_rewriter"
                }
            )
        # Компиляция
        self.graph = workflow.compile()
        self.logger.info("Workflow построен успешно")
    
    async def process_query(self, 
                            user_id: str,
                            session_id: str,
                            query: str,
                            max_iterations: int = 3) -> Dict[str, Any]:
        """Основной метод обработки запроса"""
        # Учёт конфигурации max_iterations из config, если есть
        mi = self.config.get("max_iterations")
        if isinstance(mi, int) and mi > 0:
            max_iter = mi
        else:
            max_iter = max_iterations
        
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
                "workflow_version": self.config.get("workflow_version", "1.0")
            },
            current_step="question_critic",
            max_iterations=max_iter,
            iteration_count=0,
            errors=[]
        )
        
        try:
            self.logger.info(f"[Orch] Начинаю обработку запроса: '{query}' (max_iter={max_iter})")
            final_state = await self.graph.ainvoke(initial_state)
            
            overall_conf = self._calculate_overall_confidence(final_state)
            result = {
                "success": True,
                "answer": final_state.get("final_answer"),
                "confidence": overall_conf,
                "metadata": {
                    **final_state.get("metadata", {}),
                    "end_time": datetime.now().isoformat(),
                    "total_iterations": final_state.get("iteration_count", 0),
                    "confidence_scores": final_state.get("confidence_scores", {}),
                    "errors": final_state.get("errors", [])
                }
            }
            self.logger.info(f"[Orch] Запрос обработан: итераций={final_state.get('iteration_count', 0)}, confidence={overall_conf:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"[Orch] Ошибка при обработке запроса: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "end_time": datetime.now().isoformat(),
                    "error_details": str(e)
                }
            }
    
    # Каждый узел: синхронизируем metadata из context обратно в state.metadata
    async def _question_critic_node(self, state: WorkflowState) -> WorkflowState:
        self.logger.debug("[Orch] Entering question_critic_node")
        context = self._state_to_context(state)
        result = await self.agent_manager.execute_agent("question_critic", context)
        # Результат
        state["agent_results"]["question_critic"] = result.data
        state["confidence_scores"]["question_critic"] = result.confidence
        # Если агент сохранил что-то в context.metadata, синхронизируем
        state.setdefault("metadata", {})
        state["metadata"].update(context.metadata)
        state["current_step"] = "question_critic"
        if not result.success:
            state["errors"].append(f"Question Critic: {result.error_message}")
        self.logger.debug(f"[Orch] After question_critic: data={result.data}, conf={result.confidence}, metadata_keys={list(state['metadata'].keys())}, errors={state['errors']}")
        return state
    
    async def _query_rewriter_node(self, state: WorkflowState) -> WorkflowState:
        self.logger.debug("[Orch] Entering query_rewriter_node")
        context = self._state_to_context(state)
        result = await self.agent_manager.execute_agent("query_rewriter", context)
        # Синхронизируем metadata
        state.setdefault("metadata", {})
        state["metadata"].update(context.metadata)
        # Обработка переписанного запроса
        if result.success and result.data:
            rewritten = result.data.get("rewritten_query")
            if rewritten:
                state["processed_query"] = rewritten
            else:
                state["processed_query"] = state["original_query"]
        else:
            state["processed_query"] = state["original_query"]
        state["agent_results"]["query_rewriter"] = result.data
        state["confidence_scores"]["query_rewriter"] = result.confidence
        state["current_step"] = "query_rewriter"
        if not result.success:
            state["errors"].append(f"Query Rewriter: {result.error_message}")
        self.logger.debug(f"[Orch] After query_rewriter: processed_query={state.get('processed_query')}, data={result.data}, conf={result.confidence}, errors={state['errors']}")
        return state
    
    async def _topic_classifier_node(self, state: WorkflowState) -> WorkflowState:
        self.logger.debug("[Orch] Entering topic_classifier_node")
        context = self._state_to_context(state)
        result = await self.agent_manager.execute_agent("topic_classifier", context)
        state.setdefault("metadata", {})
        state["metadata"].update(context.metadata)
        # Читаем правильное поле: primary_topic
        if result.success and result.data:
            topic = result.data.get("primary_topic") or result.data.get("topic")
            state["topic"] = topic
            # Можно сохранить полное тело классификации
            state["metadata"]["topic_classification"] = result.data
        state["agent_results"]["topic_classifier"] = result.data
        state["confidence_scores"]["topic_classifier"] = result.confidence
        state["current_step"] = "topic_classifier"
        if not result.success:
            state["errors"].append(f"Topic Classifier: {result.error_message}")
        self.logger.debug(f"[Orch] After topic_classifier: topic={state.get('topic')}, data_keys={list(result.data.keys()) if result.data else None}, conf={result.confidence}, errors={state['errors']}")
        return state
    
    async def _document_retriever_node(self, state: WorkflowState) -> WorkflowState:
        self.logger.debug("[Orch] Entering document_retriever_node")
        context = self._state_to_context(state)
        result = await self.agent_manager.execute_agent("document_retriever", context)
        state.setdefault("metadata", {})
        state["metadata"].update(context.metadata)
        # Обработка результатов
        if isinstance(result.data, list):
            docs = result.data
        elif isinstance(result.data, dict):
            docs = result.data.get("documents", [])
        else:
            docs = []
        state["documents"] = docs
        state["metadata"]["retrieved_docs"] = docs
        state["agent_results"]["document_retriever"] = result.data
        state["confidence_scores"]["document_retriever"] = result.confidence
        state["current_step"] = "document_retriever"
        if not result.success:
            state["errors"].append(f"Document Retriever: {result.error_message}")
        cnt = len(docs) if docs is not None else 0
        self.logger.debug(f"[Orch] After document_retriever: found {cnt} docs, conf={result.confidence}, errors={state['errors']}")
        return state
    
    async def _context_enricher_node(self, state: WorkflowState) -> WorkflowState:
        self.logger.debug("[Orch] Entering context_enricher_node")
        context = self._state_to_context(state)
        result = await self.agent_manager.execute_agent("context_enricher", context)
        state.setdefault("metadata", {})
        state["metadata"].update(context.metadata)
        if result.success and result.data:
            # Предполагаем, что агент возвращает ключ "enriched_context" или "enriched_documents"
            enriched = result.data.get("enriched_context") or result.data.get("enriched_documents")
            if enriched is not None:
                # Если это новые документы, обновляем state["documents"]; иначе просто сохраняем enriched_context
                if isinstance(enriched, list):
                    state["documents"] = enriched
                    state["metadata"]["enriched_docs"] = enriched
                else:
                    state["metadata"]["enriched_context"] = enriched
        state["agent_results"]["context_enricher"] = result.data
        state["confidence_scores"]["context_enricher"] = result.confidence
        state["current_step"] = "context_enricher"
        if not result.success:
            state["errors"].append(f"Context Enricher: {result.error_message}")
        docs_cnt = len(state.get("documents") or [])
        self.logger.debug(f"[Orch] After context_enricher: enriched_docs_count={docs_cnt}, conf={result.confidence}, errors={state['errors']}")
        return state
    
    async def _answer_generator_node(self, state: WorkflowState) -> WorkflowState:
        self.logger.debug("[Orch] Entering answer_generator_node")
        context = self._state_to_context(state)
        # Логируем вход
        docs_cnt = len(context.documents or [])
        self.logger.debug(f"[Orch] AnswerGenerator input: processed_query={context.processed_query}, topic={context.topic}, documents_in_context={docs_cnt}")
        result = await self.agent_manager.execute_agent("answer_generator", context)
        state.setdefault("metadata", {})
        state["metadata"].update(context.metadata)
        # Читаем корректное поле: сначала пытаемся "answer", затем "answer_text"
        ans = None
        if result.success and result.data:
            ans = result.data.get("answer") or result.data.get("answer_text")
            state["final_answer"] = ans
            # Сохраняем в metadata
            state["metadata"]["answer"] = ans
        state["agent_results"]["answer_generator"] = result.data
        state["confidence_scores"]["answer_generator"] = result.confidence
        state["current_step"] = "answer_generator"
        if not result.success:
            state["errors"].append(f"Answer Generator: {result.error_message}")
        self.logger.debug(f"[Orch] After answer_generator: final_answer_set={'yes' if state.get('final_answer') else 'no'}, conf={result.confidence}, errors={state['errors']}")
        return state
    
    async def _quality_checker_node(self, state: WorkflowState) -> WorkflowState:
        self.logger.debug("[Orch] Entering quality_checker_node")
        context = self._state_to_context(state)
        # Передаем в metadata ответ под ключом "answer"
        state.setdefault("metadata", {})
        # Убедимся, что context.metadata["answer"] установлен
        if state.get("final_answer") is not None:
            context.metadata["answer"] = state["final_answer"]
        result = await self.agent_manager.execute_agent("quality_checker", context)
        state.setdefault("metadata", {})
        state["metadata"].update(context.metadata)
        state["agent_results"]["quality_checker"] = result.data
        state["confidence_scores"]["quality_checker"] = result.confidence
        state["current_step"] = "quality_checker"
        # Увеличиваем счетчик итераций
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        if not result.success:
            state["errors"].append(f"Quality Checker: {result.error_message}")
        self.logger.debug(f"[Orch] After quality_checker: data_keys={list(result.data.keys()) if result.data else None}, conf={result.confidence}, iteration_count={state['iteration_count']}, errors={state['errors']}")
        return state
    
    def _should_retry(self, state: WorkflowState) -> str:
        """
        Определяет, нужно ли повторить обработку.
        Читает подсказки из качества: needs_context_enrichment, needs_query_improvement.
        Если retry/enhance — очищаем соответствующие поля из state для новой итерации.
        """
        quality_result = state.get("agent_results", {}).get("quality_checker", {}) or {}
        quality_score = state.get("confidence_scores", {}).get("quality_checker", 0.0)
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        
        if iteration_count >= max_iterations:
            self.logger.info(f"[Orch] Достигнуто макс итераций: {max_iterations}")
            return "finish"
        
        # Если низкое качество
        if quality_score < 0.7:
            if quality_result.get("needs_context_enrichment"):
                self.logger.info("[Orch] Решение: enhance (context enrichment)")
                # Очистка enriched контекста и ответа, но оставляем документы
                state["metadata"].pop("enriched_context", None)
                state["metadata"].pop("enriched_docs", None)
                state["metadata"].pop("answer", None)
                state["final_answer"] = None
                return "enhance"
            elif quality_result.get("needs_query_improvement"):
                self.logger.info("[Orch] Решение: retry (query improvement)")
                # Очистка документов, обогащённого контекста, ответа
                state["documents"] = None
                state["metadata"].pop("retrieved_docs", None)
                state["metadata"].pop("enriched_context", None)
                state["metadata"].pop("enriched_docs", None)
                state["metadata"].pop("answer", None)
                state["final_answer"] = None
                return "retry"
        
        # Качество приемлемое
        if quality_score >= 0.7:
            self.logger.info(f"[Orch] Качество приемлемое: {quality_score}")
            return "finish"
        
        # По умолчанию retry
        self.logger.info("[Orch] По умолчанию повторяем обработку")
        state["documents"] = None
        state["metadata"].pop("retrieved_docs", None)
        state["metadata"].pop("enriched_context", None)
        state["metadata"].pop("enriched_docs", None)
        state["metadata"].pop("answer", None)
        state["final_answer"] = None
        return "retry"
    
    def _state_to_context(self, state: WorkflowState) -> AgentContext:
        """Преобразование состояния workflow в контекст агента"""
        metadata = {
            **state.get("metadata", {}),
            "current_step": state.get("current_step"),
            "iteration_count": state.get("iteration_count", 0),
            "agent_results": state.get("agent_results", {}),
            "confidence_scores": state.get("confidence_scores", {})
        }
        return AgentContext(
            user_id=state["user_id"],
            session_id=state["session_id"],
            original_query=state["original_query"],
            processed_query=state.get("processed_query"),
            topic=state.get("topic"),
            documents=state.get("documents"),
            metadata=metadata
        )
    
    def _calculate_overall_confidence(self, state: WorkflowState) -> float:
        """Расчет общей уверенности в результате"""
        scores = state.get("confidence_scores", {})
        if not scores:
            return 0.0
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
        return {
            "session_id": session_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Получение метрик workflow (пока заглушка)"""
        return {
            "total_processed": 0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0,
            "success_rate": 0.0
        }

class WorkflowBuilder:
    """Строитель для создания различных workflow конфигураций"""
    
    @staticmethod
    def create_fast_workflow(agent_manager: AgentManager) -> DocumentationOrchestrator:
        config = {
            "workflow_type": "fast",
            "max_iterations": 1,
            "skip_quality_check": False,
            "parallel_processing": False  # параллельность пока не поддерживается
        }
        return DocumentationOrchestrator(agent_manager, config)
    
    @staticmethod
    def create_thorough_workflow(agent_manager: AgentManager) -> DocumentationOrchestrator:
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
        config = {
            "workflow_type": "debug",
            "max_iterations": 1,
            "skip_quality_check": True,
            "verbose_logging": True,
            "save_intermediate_results": True
        }
        return DocumentationOrchestrator(agent_manager, config)
