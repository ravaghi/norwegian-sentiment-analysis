from pydantic import BaseModel
from datetime import datetime


class Article(BaseModel):
    source: str
    url: str
    title: str
    content: str
    publish_date: datetime
    polarity: int
    image_urls: dict
    authors: list
