import os
from pymongo import MongoClient
from .models import Article

# MongoDB driver

client = MongoClient(os.environ.get('MONGODB_URI'))

database = client.SENA
collection = database.articles


def fetch_article(title: str) -> Article:
    """Fetch an article from the database

    Args:
        title: Title of the article to fetch

    Returns: Article object

    """
    document = collection.find_one({'title': title})
    return document


def fetch_articles() -> list:
    """Fetch all articles from the database

    Returns: List of Article objects

    """
    articles = []
    cursor = collection.find({})
    for document in cursor:
        articles.append(Article(**document))

    return articles


def save_articles(articles: list) -> list:
    """Save a list of articles to the database

    Args:
        articles: List of Article objects

    Returns: List of Article objects

    """
    return collection.insert_many(articles)


# Check if article with a given url exists in the database
def article_exists(url: str) -> bool:
    """Check if an article exists in the database

    Args:
        url: URL of the article to check

    Returns: True if article exists, False otherwise

    """
    return collection.count_documents({'url': url}) > 0
