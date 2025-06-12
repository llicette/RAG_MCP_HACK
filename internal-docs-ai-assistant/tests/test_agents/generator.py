import re
import os

from dotenv import load_dotenv
from pathlib import Path
from langchain_gigachat import GigaChat
from langchain_core.language_models import BaseChatModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


def get_model(model_name: str = "GigaChat-2") -> BaseChatModel:
    api_key = os.getenv("GIGACHAT_AUTHORIZATION_KEY")
    if not api_key:
        raise ValueError("GIGACHAT_AUTHORIZATION_KEY not found in .env")

    giga = GigaChat(
        credentials=api_key,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model=model_name
    )
    return giga


docs_dir = "documents"
pdf_files = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]

all_docs = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(os.path.join(docs_dir, pdf_file))
    docs = loader.load()
    all_docs.extend(docs)

texts = [doc.page_content for doc in all_docs]

llm = get_model()

prompt = PromptTemplate.from_template("""
Ты — помощник, который создаёт пары «вопрос–ответ» по заданному тексту.
Создай 10 правильных и неправильных вопросов на русском языке и дай на них подробные ответы, 
отвечая только цитатами из текста, либо пишеь что не знаешь точно ответа.
Чёткий формат вывода Вопрос: и Ответ:

Текст:
{text}
""")

chain = LLMChain(llm=llm, prompt=prompt)

print("🔍 Генерируем вопросы и ответы...")
response = chain.invoke({"text": texts})
output = response["text"]

print("\n✨ Результат:")
pattern = r'Вопрос: (.*?)\s*Ответ: (.*?)(?=\n|$)'
matches = re.findall(pattern, output, re.DOTALL)

for q, a in matches:
    print(f"❓ Вопрос: {q.strip()}")
    print(f"✅ Ответ: {a.strip()}")
    print("-" * 60)
