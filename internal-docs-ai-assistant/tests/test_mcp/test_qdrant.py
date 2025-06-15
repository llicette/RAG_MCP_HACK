# tests/check_qdrant_connection.py
import asyncio
from qdrant_client import QdrantClient
import os

async def main():
    url = os.getenv('QDRANT_URL')
    api_key = os.getenv('QDRANT_API_KEY')
    
    try:
        client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False)
        # Получим список коллекций
        resp = client.get_collections()
        print("Collections:", resp)
    except Exception as e:
        print("Ошибка при подключении к Qdrant:", e)

if __name__ == "__main__":
    asyncio.run(main())
