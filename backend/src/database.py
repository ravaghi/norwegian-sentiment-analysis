import os
from .models import Article

# MongoDB driver
import motor.motor_asyncio

client = motor.motor_asyncio.AsyncIOMotorClient(os.environ.get('MONGODB_URI'))

database = client.SENA
collection = database.articles


async def fetch_article(title: str) -> Article:
    document = await collection.find_one({'title': title})
    return document


async def fetch_articles() -> list:
    articles = []
    cursor = collection.find({})
    async for document in cursor:
        articles.append(Article(**document))

    return articles
