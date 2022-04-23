import newspaper
from newspaper import Article
from tqdm import tqdm
import pandas as pd


def load_news_articles() -> pd.DataFrame:
    """
    Loads news articles from the news sources.
    """
    sources = [
        'https://www.nrk.no/',
        'https://www.vg.no/',
        'https://www.aftenposten.no/',
        'https://www.nettavisen.no/'
    ]

    result = []
    for source in sources:
        articles = newspaper.build(source, language='no', memoize_articles=False)
        for article in tqdm(articles.articles, desc=f'Loading articles from {source}'):
            try:
                article = Article(article.url)
                article.download()
                article.parse()
                if article.text and article.title and source in article.url:
                    result.append({
                        'source': source.split('.')[1],
                        'url': article.url,
                        'title': article.title,
                        'text': article.text
                    })
            except Exception as e:
                continue

    return pd.DataFrame(result).sample(frac=1).reset_index(drop=True)
