from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging
from langfuse import Langfuse

@dataclass
class AgentContext:
    """Контекст для передачи между агентами"""
    user_id: str
    session_id: str
    original_query: str
    processed_query: Optional[str] = None
    topic: Optional[str] = None
    documents: Optional[List[Dict]] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AgentResult:
    """Результат выполнения агента"""
    success: bool
    data: Any
    confidence: float
    processing_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseAgent(ABC):
    """Базовый класс для всех агентов"""
    
    def __init__(self, 
                 name: str,
                 config: Dict[str, Any],
                 langfuse_client: Optional[Langfuse] = None):
        self.name = name
        self.config = config
        self.langfuse = langfuse_client
        self.logger = logging.getLogger(f"agent.{name}")
        self._setup_logger()
    
    def _setup_logger(self):
        """Настройка логгера для агента"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'%(asctime)s - {self.name} - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """Главный метод выполнения агента"""
        start_time = asyncio.get_event_loop().time()
        
        # Трейсинг с LangFuse
        trace = None
        if self.langfuse:
            trace = self.langfuse.trace(
                name=f"{self.name}_execution",
                user_id=context.user_id,
                session_id=context.session_id,
                input={"query": context.original_query}
            )
        
        try:
            self.logger.info(f"Начинаю выполнение для запроса: {context.original_query}")
            
            # Предварительная обработка
            context = await self._preprocess(context)
            
            # Основная логика агента
            result_data = await self._process(context)
            
            # Постобработка
            result_data = await self._postprocess(result_data, context)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            result = AgentResult(
                success=True,
                data=result_data,
                confidence=self._calculate_confidence(result_data, context),
                processing_time=processing_time
            )
            
            # Логирование успешного выполнения
            if trace:
                trace.update(
                    output={"result": result_data},
                    metadata={"processing_time": processing_time}
                )
            
            self.logger.info(f"Успешно выполнено за {processing_time:.2f}с")
            return result
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            error_msg = f"Ошибка в агенте {self.name}: {str(e)}"
            
            self.logger.error(error_msg, exc_info=True)
            
            if trace:
                trace.update(
                    output={"error": error_msg},
                    metadata={"processing_time": processing_time}
                )
            
            return AgentResult(
                success=False,
                data=None,
                confidence=0.0,
                processing_time=processing_time,
                error_message=error_msg
            )
    
    async def _preprocess(self, context: AgentContext) -> AgentContext:
        """Предварительная обработка (переопределяется в наследниках)"""
        return context
    
    @abstractmethod
    async def _process(self, context: AgentContext) -> Any:
        """Основная логика агента (обязательно переопределяется)"""
        pass
    
    async def _postprocess(self, result_data: Any, context: AgentContext) -> Any:
        """Постобработка результата (переопределяется в наследниках)"""
        return result_data
    
    def _calculate_confidence(self, result_data: Any, context: AgentContext) -> float:
        """Расчет уверенности в результате (переопределяется в наследниках)"""
        return 1.0
    
    async def health_check(self) -> bool:
        """Проверка состояния агента"""
        try:
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Получение конфигурационного параметра"""
        return self.config.get(key, default)
    
    def update_config(self, updates: Dict[str, Any]):
        """Обновление конфигурации"""
        self.config.update(updates)
        self.logger.info(f"Конфигурация обновлена: {updates}")

class AgentManager:
    """Менеджер для управления агентами"""
    
    def __init__(self, langfuse_client: Optional[Langfuse] = None):
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

# Декораторы для агентов
def with_retry(max_attempts: int = 3, delay: float = 1.0):
    """Декоратор для повторных попыток выполнения"""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    self.logger.warning(f"Попытка {attempt + 1} неудачна: {e}")
                    await asyncio.sleep(delay)
            return None
        return wrapper
    return decorator

def with_timeout(timeout_seconds: float = 30.0):
    """Декоратор для установки таймаута"""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(self, *args, **kwargs), 
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                raise Exception(f"Превышен таймаут {timeout_seconds}с")
        return wrapper
    return decorator