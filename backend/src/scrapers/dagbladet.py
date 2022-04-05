from json import JSONDecodeError

from newspaper import Newspaper
from datetime import datetime
import json


class Dagbladet(Newspaper):
    URL = "https://www.dagbladet.no/"
    NAME = "Dagbladet"

    def __init__(self):
        super().__init__(self.URL, self.NAME)

    @staticmethod
    def _get_article_detail(soup):
        try:
            detail = soup.find("script", type="application/ld+json")
            detail = json.loads(detail.text)
            return detail
        except (AttributeError, JSONDecodeError, TypeError):
            return None

    def _get_urls(self):
        soup = self.soup()

        result = set()
        a_tags = soup.find_all("a", {"itemprop": "url"}, href=True)
        for a_tag in a_tags:
            temp_url = a_tag["href"].strip()
            url_length = len(a_tag["href"].split("/"))
            if url_length > 3 and temp_url.startswith(self.URL) and not self.article_exists(temp_url):
                result.add(temp_url)
        return list(result)

    def _get_title(self, soup):
        try:
            return soup.find("meta", property="og:title")["content"]
        except TypeError:
            return None

    def _get_content(self, soup):
        try:
            detail = self._get_article_detail(soup)
            return detail["articleBody"]
        except (TypeError, KeyError):
            return None

    def _get_publish_date(self, soup):
        try:
            detail = self._get_article_detail(soup)
            published_date = detail["datePublished"]
            published_date = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%S.%fZ")
            return published_date
        except (TypeError, KeyError):
            return None

    def _get_image_urls(self, soup):
        try:
            article_div = soup.find("article")
            content_images = article_div.find_all("figure")

            result = []
            for image in content_images:
                caption = image.find("figcaption").text
                srcset = image.find("picture")["srcset"]
                result.append({"caption": caption, "srcset": srcset})

            return result
        except (AttributeError, TypeError, KeyError):
            return []

    def _get_authors(self, soup):
        try:
            detail = self._get_article_detail(soup)
            authors = detail["author"]
            authors = [author["name"] for author in authors if "(" not in author["name"]]
            return authors
        except (TypeError, KeyError):
            return None
