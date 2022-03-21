import os

# MongoDB driver
import motor.motor_asyncio

client = motor.motor_asyncio.AsyncIOMotorClient(os.environ.get('MONGODB_URI'))

database = client.SENA
collection = database.articles


async def fetch_article(article_id):
    article = await collection.find_one({'_id': article_id})
    return article


async def fetch_articles():
    articles = await collection.find().to_list(length=None)
    return articles


async def create_article(article):
    await collection.insert_one(article)


async def update_article(article_id, article):
    await collection.update_one({'_id': article_id}, {'$set': article})


async def delete_article(article_id):
    await collection.delete_one({'_id': article_id})
