import asyncio
from typing import Dict, Any, List

from agents.base_agent import BaseAgent, AgentContext, AgentResult, with_retry, with_timeout
from utils.vector_store_client import VectorStoreClient 
import logging
from configs.settings import settings

class RetrieverAgent(BaseAgent):
    """Агент для поиска документов через VectorStoreClient."""

    def __init__(self, config: Dict[str, Any], langfuse_client=None):
        super().__init__("document_retriever", config, langfuse_client)
        # Инициализируем VectorStoreClient.
        # Предполагается, что VectorStoreClient сам читает настройки из config или окружения.
        try:
            # Если VectorStoreClient требует передачу config, можно передать: VectorStoreClient(config)
            self.vector_client = VectorStoreClient()
        except Exception as e:
            self.logger.error(f"RetrieverAgent: не удалось инициализировать VectorStoreClient: {e}", exc_info=True)
            # Чтобы не падать при __init__, оставляем клиент = None, но далее проверяем
            self.vector_client = None

        # Параметр top_k: сколько документов возвращать
        self.top_k = int(self.get_config("top_k", 5))

        # Уровень логирования для подробностей
        self.logger.setLevel(logging.DEBUG)

    @with_timeout(60.0)
    @with_retry(max_attempts=2)
    async def _process(self, context: AgentContext) -> List[Dict[str, Any]]:
        """
        Основная логика поиска:
        - Формируем query: предпочитаем processed_query, иначе original_query.
        - Вызываем VectorStoreClient.search.
        - Ожидаем, что он возвращает List[Dict] с ключами, например: source, title, content, score.
        """
        query = ""
        if context.processed_query and isinstance(context.processed_query, str) and context.processed_query.strip():
            query = context.processed_query.strip()
        elif context.original_query and isinstance(context.original_query, str):
            query = context.original_query.strip()
        else:
            query = ""

        if not query:
            # Если нет текста запроса, возвращаем пустой список
            self.logger.warning("RetrieverAgent: пустой запрос, возвращаем пустой список документов")
            return []

        if not self.vector_client:
            raise RuntimeError("RetrieverAgent: VectorStoreClient не инициализирован")

        self.logger.debug(f"RetrieverAgent: начинаем поиск по запросу: '{query}', top_k={self.top_k}")
        try:
            # Предполагаем, что search — асинхронный метод: await self.vector_client.search(...)
            docs = await self.vector_client.search(query, top_k=self.top_k)
            # Проверим, что docs — список
            if not isinstance(docs, list):
                self.logger.warning("RetrieverAgent: VectorStoreClient.search вернул не список, приводим к пустому")
                return []
            # Каждый элемент предполагаем dict с ключами: source, title, content, score (или похожими)
            # При необходимости можно выполнить нормализацию: гарантировать наличие полей.
            normalized_docs: List[Dict[str, Any]] = []
            for idx, doc in enumerate(docs):
                if not isinstance(doc, dict):
                    self.logger.debug(f"RetrieverAgent: элемент результата поиска с индексом {idx} не dict, пропускаем")
                    continue
                # Нормализация полей:
                source = doc.get("source") or doc.get("id") or f"doc_{idx}"
                title = doc.get("title") or source
                content = doc.get("content") or doc.get("snippet") or ""
                # score может отсутствовать
                score = doc.get("score")
                try:
                    score = float(score) if score is not None else None
                except:
                    score = None
                normalized = {
                    "source": source,
                    "title": title,
                    "content": content,
                }
                if score is not None:
                    normalized["score"] = score
                normalized_docs.append(normalized)
            self.logger.debug(f"RetrieverAgent: найдено документов: {len(normalized_docs)}")
            return normalized_docs

        except Exception as e:
            self.logger.error(f"RetrieverAgent: ошибка при поиске: {e}", exc_info=True)
            # В случае ошибки возвращаем пустой список или бросаем: здесь возвращаем пустой, чтобы остальные агенты могли продолжить
            return []

    async def _postprocess(self, result_data: List[Dict[str, Any]], context: AgentContext) -> List[Dict[str, Any]]:
        """
        Сохраняем результаты поиска в context.metadata, чтобы другие агенты могли использовать.
        Обычно ключ: 'retrieved_docs'.
        """
        # Сохраняем normalized_docs
        context.metadata["retrieved_docs"] = result_data
        # Для совместимости можно ещё записать в context.documents
        context.documents = result_data
        return result_data

    def _calculate_confidence(self, result_data: Any, context: AgentContext) -> float:
        """
        Рассчитываем уверенность агента:
        - Если есть хотя бы один документ, уверенность высокая, например 0.9.
        - Если нет документов, уверенность низкая, например 0.2.
        """
        if isinstance(result_data, list) and result_data:
            # Можно учитывать средний или максимальный score, если есть
            scores = [doc.get("score") for doc in result_data if isinstance(doc.get("score"), (int, float))]
            if scores:
                # Нормализуем: предположим, score от 0 до 1; средний + бонус
                try:
                    avg = sum(scores) / len(scores)
                    # Привести к [0.5, 1.0]
                    conf = 0.5 + 0.5 * avg
                    return min(max(conf, 0.5), 1.0)
                except:
                    return 0.9
            return 0.9
        # Нет результатов
        return 0.2


def create_retriever_agent(config: Dict[str, Any] = None, langfuse_client=None) -> RetrieverAgent:
    """
    Фабрика для создания экземпляра RetrieverAgent.
    При необходимости config может содержать 'top_k' или прочие настройки для VectorStoreClient.
    """
    default_config = {
        "top_k": 5,
        # Можно добавить ключи конфигурации для VectorStoreClient, если он умеет читать их из config
        # Например: "vector_index_name": "documents", и т.п.
    }
    if config:
        default_config.update(config)
    return RetrieverAgent(default_config, langfuse_client)
