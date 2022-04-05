from .newspaper import Newspaper
from datetime import datetime


class NRK(Newspaper):
    URL = "https://www.nrk.no/"
    NAME = "NRK"

    def __init__(self):
        super().__init__(self.URL, self.NAME)

    def _get_urls(self):
        soup = self.soup()

        article_urls = set()
        for data in soup.find("main"):  # Check only the main tag
            tag = data.name
            if tag == "section":  # NRK divides their home page into section tags
                section_urls = data.find_all("a", href=True)
                for section_url in section_urls:
                    temp_url = section_url["href"].strip()
                    url_length = len(section_url["href"].split("/"))
                    if url_length > 4 and temp_url.startswith(self.URL) and not self.article_exists(temp_url):
                        article_urls.add(temp_url)

        return list(article_urls)

    def _get_title(self, soup):
        try:
            return soup.find("meta", property="og:title")["content"]
        except TypeError:
            return None

    def _get_content(self, soup):
        try:
            content_div = soup.find("div", {"class": "article-body"})
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
            date = soup.find("time", {"class": "datetime-absolute datePublished"})["datetime"]
            date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S%z")
            return date
        except TypeError:
            return None

    def _get_image_urls(self, soup):
        try:
            content_div = soup.find("div", {"class": "article-body"})
            content_images = content_div.find_all("figure")

            result = []
            for image in content_images:
                caption = image.find("figcaption").find("p").text
                srcset = image.find("img")["srcset"]
                result.append({"caption": caption, "srcset": srcset})

            return result
        except (AttributeError, TypeError, KeyError):
            return []

    def _get_authors(self, soup):
        try:
            authors = soup.find_all("a", {"class": "author__name"})
            return [author.text for author in authors]
        except TypeError:
            return None
