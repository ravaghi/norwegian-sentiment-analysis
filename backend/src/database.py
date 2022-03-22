import os
from .models import Article

# MongoDB driver
import motor.motor_asyncio

client = motor.motor_asyncio.AsyncIOMotorClient(os.environ.get('MONGODB_URI'))

database = client.SENA
collection = database.articles


async def fetch_article(article_id):
    document = await collection.find_one({'_id': article_id})
    return document


async def fetch_articles():
    articles = []

    cursor = collection.find({})
    async for document in cursor:
        articles.append(Article(**document))

    return articles


async def create_article(article):
    document = article
    result = await collection.insert_one(document)
    return document


async def update_article(article_id, article):
    await collection.update_one({'_id': article_id}, {'$set': article})
    document = await collection.find_one({'_id': article_id})
    return document


async def delete_article(article_id):
    await collection.delete_one({'_id': article_id})
    return True
