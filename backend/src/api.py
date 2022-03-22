from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import Article
from .database import (
    fetch_articles,
    fetch_article,
    create_article,
    update_article,
    delete_article
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
async def get_articles():
    response = await fetch_articles()
    return response


@app.get("/articles/{article_id}", response_model=Article)
async def get_article(article_id: int):
    response = await fetch_article(article_id)
    if response:
        return response
    else:
        raise HTTPException(status_code=404, detail="Article not found")


@app.post("/articles", response_model=Article)
async def post_article(article: Article):
    response = await create_article(article.dict())
    if response:
        return response
    else:
        raise HTTPException(status_code=400, detail="Article not created")


@app.put("/articles/{article_id}", response_model=Article)
async def put_article(article_id: int, data: Article):
    response = await update_article(article_id, data.dict())
    if response:
        return response
    else:
        raise HTTPException(status_code=404, detail="Article not found")


@app.delete("/articles/{article_id}")
async def delete_article(article_id: int):
    response = await delete_article(article_id)
    if response:
        return response
    else:
        raise HTTPException(status_code=404, detail="Article not found")
