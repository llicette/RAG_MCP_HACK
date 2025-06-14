import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from datetime import datetime
from configs.settings import settings
from langfuse import Langfuse
import os
@dataclass
class AgentContext:
    """Контекст для передачи между агентами."""
    user_id: str
    session_id: str
    original_query: str
    processed_query: Optional[str] = None
    topic: Optional[str] = None
    documents: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    

@dataclass
class AgentResult:
    """Результат выполнения агента."""
    success: bool
    data: Any = None
    confidence: float = 0.0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Декораторы retry и timeout
def with_retry(max_attempts: int = 2, exceptions: tuple = (Exception,)):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                last_exc = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exc = e
                        if attempt < max_attempts:
                            wait = 2 ** (attempt - 1)
                            logging.getLogger("agent").warning(
                                f"{func.__name__} attempt {attempt} failed: {e}, retrying in {wait}s"
                            )
                            await asyncio.sleep(wait)
                        else:
                            logging.getLogger("agent").error(
                                f"{func.__name__} failed after {attempt} attempts: {e}"
                            )
                # После всех попыток
                raise last_exc
            return wrapper
        else:
            return func
    return decorator

def with_timeout(timeout: float):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                except asyncio.TimeoutError as e:
                    logging.getLogger("agent").error(f"{func.__name__} timed out after {timeout}s")
                    raise
            return wrapper
        else:
            return func
    return decorator

class BaseAgent(ABC):
    """Базовый класс для всех агентов."""
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, langfuse_client: Any = None):
        self.name = name
        self.config = config or {}
        self.llm = None  # Подкласс должен инициализировать self.llm с методом invoke(prompt)->str
        self.logger = logging.getLogger(f"agent.{self.name}")
        self.langfuse = langfuse_client

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    async def invoke_llm(self, prompt: str) -> str:
        """
        Унифицированный вызов LLM с retry/backoff и таймаутом.
        """
        async def call():
            return await asyncio.to_thread(self.llm.invoke, prompt)

        last_exc = None
        for attempt in range(1, settings.LLM_RETRY_ATTEMPTS + 1):
            try:
                return await asyncio.wait_for(call(), timeout=settings.LLM_TIMEOUT)
            except Exception as e:
                last_exc = e
                if attempt < settings.LLM_RETRY_ATTEMPTS:
                    wait = 2 ** (attempt - 1)
                    self.logger.warning(f"LLM call attempt {attempt} failed: {e}, retrying in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    self.logger.error(f"LLM call permanently failed after {attempt} attempts: {e}")
        # Если все неудачны
        raise last_exc

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
            else:
                return {"result": parsed}
        except json.JSONDecodeError:
            sample = response.strip().replace("\n", " ")[:200]
            self.logger.warning(f"JSON parse error from LLM: {sample}...")
            return {"raw_text": response}

    async def run(self, context: AgentContext) -> AgentResult:
        """
        Публичный метод запуска агента. Оборачивает _preprocess, _process, _postprocess, логирование, тайминги.
        """
        start = asyncio.get_event_loop().time()
        # Опционально: LangFuse trace start
        lf_trace = None
        if self.langfuse:
            try:
                lf_trace = self.langfuse.trace(
                    name=f"{self.name}_execution",
                    user_id=context.user_id,
                    session_id=context.session_id,
                    input={"query": context.original_query}
                )
            except Exception:
                lf_trace = None

        try:
            self.logger.info(f"{self.name}: start for query: {context.original_query}")
            # Preprocess
            context = await self._preprocess(context)
            # Main
            result_data = await self._process(context)
            # Postprocess
            result_data = await self._postprocess(result_data, context)
            duration = asyncio.get_event_loop().time() - start
            confidence = 0.0
            if isinstance(result_data, dict) and "confidence" in result_data:
                try:
                    confidence = float(result_data.get("confidence", 0.0))
                except:
                    confidence = 0.0
            else:
                confidence = 1.0
            result = AgentResult(
                success=True,
                data=result_data,
                confidence=confidence,
                processing_time=duration
            )
            # LangFuse trace update
            if lf_trace:
                try:
                    lf_trace.update(output={"result": result_data}, metadata={"processing_time": duration})
                except Exception:
                    pass
            self.logger.info(f"{self.name}: success in {duration:.2f}s")
            return result
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start
            err_msg = f"{self.name} error: {e}"
            self.logger.error(err_msg, exc_info=True)
            if lf_trace:
                try:
                    lf_trace.update(output={"error": err_msg}, metadata={"processing_time": duration})
                except:
                    pass
            return AgentResult(
                success=False,
                data=None,
                confidence=0.0,
                processing_time=duration,
                error_message=str(e)
            )

    async def _preprocess(self, context: AgentContext) -> AgentContext:
        """Опциональная предварительная обработка."""
        return context

    @abstractmethod
    async def _process(self, context: AgentContext) -> Any:
        """Основная логика агента: вернуть dict или список."""
        pass

    async def _postprocess(self, result_data: Any, context: AgentContext) -> Any:
        """Опциональная постобработка результата."""
        return result_data

    def _calculate_confidence(self, result_data: Any, context: AgentContext) -> float:
        """Можно переопределить, если нужен особый подсчёт."""
        if isinstance(result_data, dict) and "confidence" in result_data:
            try:
                return float(result_data.get("confidence", 0.0))
            except:
                return 0.0
        return 1.0

    async def health_check(self) -> bool:
        """Проверка готовности агента."""
        return True

    def update_config(self, updates: Dict[str, Any]):
        self.config.update(updates)
        self.logger.info(f"{self.name}: config updated: {updates}")
        
class AgentManager:
    """Менеджер для управления агентами"""
    
    def __init__(self, langfuse_client: Optional[Langfuse] = None):
        self.langfuse_client = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST")
        )
        self.agents: Dict[str, BaseAgent] = {}
        self.langfuse = langfuse_client
        self.logger = logging.getLogger("agent_manager")
    
    def register_agent(self, agent: BaseAgent):
        """Регистрация агента"""
        self.agents[agent.name] = agent
        self.logger.info(f"Агент {agent.name} зарегистрирован")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Получение агента по имени"""
        return self.agents.get(name)
    
    async def execute_agent(self, agent_name: str, context: AgentContext) -> AgentResult:
        """Выполнение конкретного агента"""
        agent = self.get_agent(agent_name)
        if not agent:
            return AgentResult(
                success=False,
                data=None,
                confidence=0.0,
                processing_time=0.0,
                error_message=f"Агент {agent_name} не найден"
            )
        
        return await agent.execute(context)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Проверка состояния всех агентов"""
        results = {}
        for name, agent in self.agents.items():
            results[name] = await agent.health_check()
        return results
    
    def list_agents(self) -> List[str]:
        """Список всех зарегистрированных агентов"""
        return list(self.agents.keys())