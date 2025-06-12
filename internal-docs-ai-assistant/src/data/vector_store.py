from langchain_huggingface import HuggingFaceEmbeddings  # или langchain.embeddings в старых версиях
from langchain_chroma import Chroma  # или langchain.vectorstores.Chroma в старых версиях
from langchain.schema import Document
import os

# 1) Задаём директорию для хранения embedding-индекса
persist_directory = "chroma_db"  # или любой другой путь, например "./db/chroma"
os.makedirs(persist_directory, exist_ok=True)

# 2) Инициализируем embeddings
# Выберите ту модель, которая у вас загружена. Например:
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3) Подключаемся к Chroma
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name="my_collection"  # имя коллекции; если её нет, Chroma создаст новую
)

# 4) Проверяем, есть ли уже документы в коллекции
# В langchain-chroma можно попытаться получить количество элементов, но API может отличаться.
# Например, если Chroma предоставляет метод get or count:
try:
    # Если есть метод .count(), используем его
    count = vectorstore._collection.count()  # внутренний доступ; в зависимости от версии API
    print(f"В коллекции ‘my_collection’ сейчас документов: {count}")
except Exception:
    print("Невозможно получить точное количество через публичный API, но коллекция подключена.")

# 5) Добавление документов
docs = [
    Document(page_content="Процедура оформления отпуска: шаги ...", metadata={"source": "doc1", "title": "Оформление отпуска"}),
    Document(page_content="Правила отпуска для сотрудников: ...", metadata={"source": "doc2", "title": "Правила отпуска"}),
]
vectorstore.add_documents(docs)
# После добавления Chroma автоматически сохранит embedding-данные в persist_directory.

# 6) Явное сохранение (для старых версий Chroma/LangChain):
try:
    vectorstore.persist()
    print("Документы добавлены и сохранены.")
except AttributeError:
    # В новых версиях Chroma авто-сохранение, persist() может не требоваться
    print("Документы добавлены; persist() либо не нужен, либо недоступен.")

# 7) Тестовый поиск: убедимся, что запрос возвращает ожидаемые документы
query = "оформление отпуска сотрудников"
results = vectorstore.similarity_search_with_score(query, k=2)
for doc, score in results:
    print(f"Найден: source={doc.metadata.get('source')}, title={doc.metadata.get('title')}, score={score}")
