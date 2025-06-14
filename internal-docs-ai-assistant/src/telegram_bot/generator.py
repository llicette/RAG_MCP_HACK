import re
import os
import json

from dotenv import load_dotenv
from pathlib import Path
from langchain_gigachat import GigaChat
from langchain_core.language_models import BaseChatModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser


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

def generate(pdf_files):
    all_docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        all_docs.extend(docs)

    texts = [doc.page_content for doc in all_docs]

    llm = get_model()

    prompt = PromptTemplate.from_template("""
    Ты — помощник, который создаёт пары «вопрос–ответ» по заданному тексту.
    Создай 5 правильных и неправильных вопросов на русском языке и дай на них подробные ответы, 
    отвечая только цитатами из текста, либо пишешь что не знаешь точно ответа.
    
    Возвращай ответы строго в виде json в соответствии с примером ниже без лишних комментариев и в
    едином формате:
    СТРОГО СТРУКТУРА JSON
    [   
        {{
            "question": "...",
            "answer": "..."
        }},
        {{
            "question": "...",
            "answer": "..."
        }}
    ]
    поле question - вопрос
    поле answer - ответ на вопрос
    СТРОГО СТРУКТУРА JSON
    
    Текст:
    {text}
    """)

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"text": texts})
    print(response)
    output = json.loads(response)

    return output

# в запросе убрать html форматирование