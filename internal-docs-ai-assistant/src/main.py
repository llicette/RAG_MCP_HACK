
"""
AI Documentation Assistant - Main Application Entry Point
Главная точка входа для системы AI Documentation Assistant
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent))

from src.core.config import AppConfig, load_config
from src.core.orchestrator import DocumentationOrchestrator, WorkflowBuilder
from src.agents.base_agent import AgentManager
from src.interfaces.telegram_bot import TelegramBotInterface
from src.mcp.server import MCPServer
from src.data.vector_store import VectorStoreManager
from src.monitoring.langfuse_client import LangFuseClient
from src.core.cache_manager import CacheManager
from src.utils.llm_clients import LLMClientManager

class DocumentationAssistantApp:
    """Главное приложение AI Documentation Assistant"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.logger = self._setup_logging()
        
        # Основные компоненты
        self.cache_manager: Optional[CacheManager] = None
        self.llm_client_manager: Optional[LLMClientManager] = None
        self.vector_store_manager: Optional[VectorStoreManager] = None
        self.agent_manager: Optional[AgentManager] = None
        self.orchestrator: Optional[DocumentationOrchestrator] = None
        self.mcp_server: Optional[MCPServer] = None
        self.telegram_bot: Optional[TelegramBotInterface] = None
        self.langfuse_client: Optional[LangFuseClient] = None
        
        # Флаги состояния
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
    def _setup_logging(self) -> logging.Logger:
        """Настройка системы логирования"""
        log_level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        
        # Настройка форматирования
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Настройка обработчиков
        handlers = []
        
        # Консольный вывод
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
        
        # Файловый вывод
        if self.config.logging.file_path:
            log_dir = Path(self.config.logging.file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(self.config.logging.file_path)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Настройка root logger
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            force=True
        )
        
        # Настройка специфичных логгеров
        for logger_name in ['orchestrator', 'agents', 'telegram_bot', 'mcp_server']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(log_level)
        
        return logging.getLogger("main_app")
    
    async def initialize(self):
        """Инициализация всех компонентов приложения"""
        self.logger.info("Инициализация AI Documentation Assistant...")
        
        try:
            # 1. Инициализация кэш-менеджера
            self.logger.info("Инициализация кэш-менеджера...")
            self.cache_manager = CacheManager(self.config.cache)
            await self.cache_manager.initialize()
            
            # 2. Инициализация LLM клиентов
            self.logger.info("Инициализация LLM клиентов...")
            self.llm_client_manager = LLMClientManager(self.config.llm)
            await self.llm_client_manager.initialize()
            
            # 3. Инициализация векторного хранилища
            self.logger.info("Инициализация векторного хранилища...")
            self.vector_store_manager = VectorStoreManager(self.config.vector_store)
            await self.vector_store_manager.initialize()
            
            # 4. Инициализация менеджера агентов
            self.logger.info("Инициализация агентов...")
            self.agent_manager = AgentManager(
                llm_client_manager=self.llm_client_manager,
                vector_store_manager=self.vector_store_manager,
                cache_manager=self.cache_manager,
                config=self.config.agents
            )
            await self.agent_manager.initialize()
            
            # 5. Создание оркестратора
            self.logger.info("Создание оркестратора...")
            if self.config.app.workflow_type == "fast":
                self.orchestrator = WorkflowBuilder.create_fast_workflow(self.agent_manager)
            elif self.config.app.workflow_type == "thorough":
                self.orchestrator = WorkflowBuilder.create_thorough_workflow(self.agent_manager)
            elif self.config.app.workflow_type == "debug":
                self.orchestrator = WorkflowBuilder.create_debug_workflow(self.agent_manager)
            else:
                self.orchestrator = DocumentationOrchestrator(self.agent_manager, {})
            
            # 6. Инициализация мониторинга
            if self.config.monitoring.langfuse.enabled:
                self.logger.info("Инициализация LangFuse...")
                self.langfuse_client = LangFuseClient(self.config.monitoring.langfuse)
                await self.langfuse_client.initialize()
            
            # 7. Инициализация MCP сервера
            if self.config.mcp.enabled:
                self.logger.info("Инициализация MCP сервера...")
                self.mcp_server = MCPServer(
                    orchestrator=self.orchestrator,
                    vector_store_manager=self.vector_store_manager,
                    config=self.config.mcp
                )
                await self.mcp_server.initialize()
            
            # 8. Инициализация Telegram бота
            if self.config.telegram.enabled:
                self.logger.info("Инициализация Telegram бота...")
                self.telegram_bot = TelegramBotInterface(
                    orchestrator=self.orchestrator,
                    cache_manager=self.cache_manager,
                    config=self.config.telegram
                )
                await self.telegram_bot.initialize()
            
            self.logger.info("Все компоненты успешно инициализированы")
            
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации: {e}", exc_info=True)
            await self.cleanup()
            raise
    
    async def start(self):
        """Запуск приложения"""
        if self.is_running:
            self.logger.warning("Приложение уже запущено")
            return
        
        self.logger.info("Запуск AI Documentation Assistant...")
        
        try:
            # Запуск компонентов
            tasks = []
            
            # Запуск MCP сервера
            if self.mcp_server:
                tasks.append(asyncio.create_task(self.mcp_server.start()))
                self.logger.info("MCP сервер запущен")
            
            # Запуск Telegram бота
            if self.telegram_bot:
                tasks.append(asyncio.create_task(self.telegram_bot.start()))
                self.logger.info("Telegram бот запущен")
            
            self.is_running = True
            self.logger.info("✅ AI Documentation Assistant успешно запущен")
            
            # Ожидание сигнала завершения
            await self.shutdown_event.wait()
            
            # Завершение задач
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
        except Exception as e:
            self.logger.error(f"Ошибка при запуске: {e}", exc_info=True)
            raise
        finally:
            await self.cleanup()
    
    async def stop(self):
        """Остановка приложения"""
        if not self.is_running:
            return
        
        self.logger.info("Остановка AI Documentation Assistant...")
        
        # Остановка компонентов
        if self.telegram_bot:
            await self.telegram_bot.stop()
            self.logger.info("Telegram бот остановлен")
        
        if self.mcp_server:
            await self.mcp_server.stop()
            self.logger.info("MCP сервер остановлен")
        
        self.is_running = False
        self.shutdown_event.set()
        
        self.logger.info("AI Documentation Assistant остановлен")
    
    async def cleanup(self):
        """Очистка ресурсов"""
        self.logger.info("Очистка ресурсов...")
        
        try:
            # Очистка в обратном порядке инициализации
            if self.telegram_bot:
                await self.telegram_bot.cleanup()
            
            if self.mcp_server:
                await self.mcp_server.cleanup()
            
            if self.langfuse_client:
                await self.langfuse_client.cleanup()
            
            if self.agent_manager:
                await self.agent_manager.cleanup()
            
            if self.vector_store_manager:
                await self.vector_store_manager.cleanup()
            
            if self.llm_client_manager:
                await self.llm_client_manager.cleanup()
            
            if self.cache_manager:
                await self.cache_manager.cleanup()
            
            self.logger.info("Ресурсы очищены")
            
        except Exception as e:
            self.logger.error(f"Ошибка при очистке ресурсов: {e}", exc_info=True)
    
    def setup_signal_handlers(self):
        """Настройка обработчиков сигналов"""
        def signal_handler(signum, frame):
            self.logger.info(f"Получен сигнал {signum}, начинаю остановку...")
            asyncio.create_task(self.stop())
        
        # Обработка сигналов завершения
        if sys.platform != "win32":
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка состояния системы"""
        status = {
            "status": "healthy" if self.is_running else "stopped",
            "timestamp": asyncio.get_event_loop().time(),
            "components": {}
        }
        
        # Проверка компонентов
        if self.cache_manager:
            status["components"]["cache"] = await self.cache_manager.health_check()
        
        if self.vector_store_manager:
            status["components"]["vector_store"] = await self.vector_store_manager.health_check()
        
        if self.llm_client_manager:
            status["components"]["llm_clients"] = await self.llm_client_manager.health_check()
        
        if self.agent_manager:
            status["components"]["agents"] = await self.agent_manager.health_check()
        
        if self.telegram_bot:
            status["components"]["telegram_bot"] = await self.telegram_bot.health_check()
        
        if self.mcp_server:
            status["components"]["mcp_server"] = await self.mcp_server.health_check()
        
        return status

# CLI интерфейс
async def main():
    """Главная функция запуска"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Documentation Assistant")
    parser.add_argument("--config", "-c", help="Путь к файлу конфигурации")
    parser.add_argument("--debug", "-d", action="store_true", help="Включить отладочный режим")
    parser.add_argument("--health-check", action="store_true", help="Проверить состояние системы")
    
    args = parser.parse_args()
    
    # Создание приложения
    app = DocumentationAssistantApp(config_path=args.config)
    
    if args.debug:
        app.config.logging.level = "DEBUG"
        app.config.app.workflow_type = "debug"
    
    try:
        # Инициализация
        await app.initialize()
        
        # Проверка состояния
        if args.health_check:
            health = await app.health_check()
            print(f"Health Status: {health}")
            return
        
        # Настройка обработчиков сигналов
        app.setup_signal_handlers()
        
        # Запуск
        await app.start()
        
    except KeyboardInterrupt:
        app.logger.info("Получен сигнал прерывания")
    except Exception as e:
        app.logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await app.cleanup()

if __name__ == "__main__":
    # Настройка политики событийного цикла для Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограмма завершена пользователем")
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)