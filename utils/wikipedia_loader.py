"""Load Wikipedia articles from local JSONL file instead of HF API."""

import json
from pathlib import Path

DEFAULT_ARTICLES_PATH = Path("wikipedia_en.jsonl")


def load_articles_by_id(
    article_ids: set[int],
    source_path: Path | None = None,
) -> dict[int, str]:
    path = source_path or DEFAULT_ARTICLES_PATH
    texts: dict[int, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            article = json.loads(line)
            aid = article["article_id"]
            if aid in article_ids:
                texts[aid] = article["text"]
                if len(texts) >= len(article_ids):
                    break
    return texts


def load_articles(
    n: int,
    source_path: Path | None = None,
) -> list[dict]:
    path = source_path or DEFAULT_ARTICLES_PATH
    articles: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            article = json.loads(line)
            articles.append({
                "article_id": article["article_id"],
                "title": article["title"],
                "text": article["text"],
            })
            if len(articles) >= n:
                break
    return articles
