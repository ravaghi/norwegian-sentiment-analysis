from .newspaper import Newspaper
from datetime import datetime
import json


class Aftenposten(Newspaper):
    URL = "https://www.aftenposten.no/"
    NAME = "Aftenposten"

    def __init__(self):
        super().__init__(self.URL, self.NAME)

    @staticmethod
    def _get_article_detail(soup):
        try:
            detail = soup.find("script", type="application/ld+json")
            detail = json.loads(detail.text)
            return detail[0]
        except AttributeError:
            return None

    def _get_urls(self):
        soup = self.soup()

        result = set()
        article_tags = soup.find_all("article")
        for article_tag in article_tags:
            try:
                temp_url = article_tag.find("a")["href"]
                temp_url = temp_url.strip()
                url_length = len(temp_url.split("/"))
                if url_length > 3 and temp_url.startswith(self.URL) and not self.article_exists(temp_url):
                    result.add(temp_url)
            except (AttributeError, TypeError):
                continue

        return list(result)

    def _get_title(self, soup):
        try:
            return soup.find("meta", property="og:title")["content"]
        except TypeError:
            return None

    def _get_content(self, soup):
        try:
            content_div = soup.find("div", id="main")
            tags = content_div.findChildren(recursive=False)
            content = ""
            for tag in tags:
                if tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    content = content + f"\n[{tag.text.strip()}]\n"
                if tag.name == "p":
                    content = content + tag.text.strip() + "\n\n"
            return content
        except AttributeError:
            return None

    def _get_publish_date(self, soup):
        try:
            detail = self._get_article_detail(soup)
            published_date = detail["datePublished"]
            published_date = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%SZ")
            return published_date
        except TypeError:
            return None

    def _get_image_urls(self, soup):
        try:
            content_div = soup.find("div", id="main")
            content_images = content_div.find_all("figure")

            result = []
            for image in content_images:
                caption = image.find("figcaption").text
                srcset = image.find("img")["srcset"]
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
        except TypeError:
            return None
