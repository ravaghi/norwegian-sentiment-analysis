import os
from .models import Article

# MongoDB driver
import motor.motor_asyncio

client = motor.motor_asyncio.AsyncIOMotorClient(os.environ.get('MONGODB_URI'))

database = client.SENA
collection = database.articles


async def fetch_article(title: str) -> Article:
    """Fetch an article from the database

    Args:
        title: Title of the article to fetch

    Returns: Article object

    """
    document = await collection.find_one({'title': title})
    return document


async def fetch_articles() -> list:
    """Fetch all articles from the database

    Returns: List of Article objects

    """
    articles = []
    cursor = collection.find({})
    async for document in cursor:
        articles.append(Article(**document))

    return articles


async def save_articles(articles: list) -> list:
    """Save a list of articles to the database

    Args:
        articles: List of Article objects

    Returns: List of Article objects

    """
    return await collection.insert_many(articles)
