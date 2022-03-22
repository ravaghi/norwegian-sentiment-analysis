from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import Article
from .database import (
    fetch_articles,
    fetch_article
)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def read_root():
    return {"Message": "Hi Mom!"}


@app.get("/articles")
async def get_articles() -> list:
    response = await fetch_articles()
    return response


@app.get("/articles/{article_id}", response_model=Article)
async def get_article(title: str) -> Article:
    response = await fetch_article(title)
    if response:
        return response
    else:
        raise HTTPException(status_code=404, detail="Article not found")
