from bs4 import BeautifulSoup
import requests
from pydantic import ValidationError
from tqdm import tqdm
from backend.src.data_collection.database import save_articles, article_exists
from backend.src.data_collection.models import Article
from abc import ABC, abstractmethod
from datetime import datetime


class Newspaper(ABC):

    @abstractmethod
    def __init__(self, url, name):
        self.url = url
        self.name = name

    def soup(self, url=None) -> BeautifulSoup:
        """Returns the soup of the url.

        Args:
            url: Article url.

        Returns: BeautifulSoup object.

        """
        if url is None:
            url = self.url
        request = requests.request("GET", url)
        soup = BeautifulSoup(request.content, "html.parser")
        return soup

    @abstractmethod
    def _get_urls(self) -> list:
        """Scrapes article urls from the news website.

        Returns: List of urls.

        """
        raise NotImplementedError

    @abstractmethod
    def _get_title(self, soup) -> str:
        """Scrapes the title of an article.

        Args:
            soup: BeautifulSoup object.

        Returns: Article title.

        """
        raise NotImplementedError

    @abstractmethod
    def _get_content(self, soup) -> str:
        """Returns the content of an article.

        Args:
            soup: BeautifulSoup object.

        Returns: Article content.

        """
        raise NotImplementedError

    @abstractmethod
    def _get_publish_date(self, soup) -> datetime:
        """Returns the published date of an article.

        Args:
            soup: BeautifulSoup object.

        Returns: Published date.

        """
        raise NotImplementedError

    @abstractmethod
    def _get_image_urls(self, soup) -> list:
        """Returns the image urls of an article.

        Args:
            soup: BeautifulSoup object.

        Returns: List of image urls.

        """
        raise NotImplementedError

    @abstractmethod
    def _get_authors(self, soup) -> list:
        """Returns the authors of an article.

        Args:
            soup: BeautifulSoup object.

        Returns: List of authors.

        """
        raise NotImplementedError

    @staticmethod
    def article_exists(url) -> bool:
        """Checks if an article exists in the database.

        Args:
            url: Article url.

        Returns: True if article exists, False otherwise.

        """
        return article_exists(url)

    def scrape(self) -> list:
        """Scrapes articles form the newspaper.

        Returns: List of articles.

        """
        urls = self._get_urls()
        articles = []
        for url in tqdm(urls, desc=f"Scraping {self.name}"):
            source = self.name
            soup = self.soup(url)
            title = self._get_title(soup)
            content = self._get_content(soup)
            publish_date = self._get_publish_date(soup)
            image_urls = self._get_image_urls(soup)
            authors = self._get_authors(soup)

            if title:
                title.strip()

            if content:
                content.strip()

            try:
                article = Article(source=source,
                                  url=url,
                                  title=title,
                                  content=content,
                                  publish_date=publish_date,
                                  polarity=-1,
                                  image_urls=image_urls,
                                  authors=authors)

                articles.append(article.dict())
            except ValidationError:
                pass

        return articles

    @staticmethod
    def save(articles):
        """Saves articles to the database.

        Args:
            articles: List of articles.

        """
        if articles:
            return save_articles(articles)
