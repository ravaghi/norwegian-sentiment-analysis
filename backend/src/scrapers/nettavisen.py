from newspaper import Newspaper
from datetime import datetime
import json


class Nettavisen(Newspaper):
    URL = "https://www.nettavisen.no/"
    NAME = "Nettavisen"

    def __init__(self):
        super().__init__(self.URL, self.NAME)

    @staticmethod
    def _get_article_detail(soup):
        try:
            detail = soup.find("script", type="application/ld+json")
            detail = json.loads(detail.text)
            return detail
        except AttributeError:
            return None

    def _get_urls(self):
        soup = self.soup()

        result = set()
        article_tags = soup.find_all("article")
        for article_tag in article_tags:
            temp_url = self.URL[:-1] + article_tag.find("a")["href"]
            temp_url = temp_url.strip()
            url_length = len(temp_url.split("/"))
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
            article_div = soup.find("article")
            article_paragraphs = article_div.find_all("p")
            content = ""
            for paragraph in article_paragraphs:
                content = content + " " + paragraph.text.strip()
            return content
        except AttributeError:
            return None

    def _get_publish_date(self, soup):
        try:
            detail = self._get_article_detail(soup)
            published_date = detail["datePublished"]
            published_date = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%S.%f%z")
            return published_date
        except TypeError:
            return None

    def _get_image_urls(self, soup):
        try:
            article_tag = soup.find("article")
            article_images = article_tag.find_all("amedia-image")

            result = []
            for image in article_images:
                caption = image.find("figcaption").text
                srcset = image.find("img")["src"]
                result.append({"caption": caption, "src": srcset})

            return result
        except (AttributeError, TypeError, KeyError):
            return []

    def _get_authors(self, soup):
        try:
            detail = self._get_article_detail(soup)
            authors = [detail["author"]]
            return authors
        except TypeError:
            return None
