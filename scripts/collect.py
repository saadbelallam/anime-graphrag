"""Fetch the most popular anime from AniList into data/raw/.

    python scripts/collect.py        # 30 pages = 1500 titles
    python scripts/collect.py 3      # quick test run
"""

import json
import os
import sys
import time

import requests

from graphrag.config import RAW_DIR

API_URL = "https://graphql.anilist.co"
DELAY = 2.5  # AniList allows 90 req/min and bans aggressive clients

QUERY = """
query ($page: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    media(type: ANIME, sort: POPULARITY_DESC, isAdult: false) {
      id
      title { romaji english }
      synonyms
      description(asHtml: false)
      seasonYear
      format
      genres
      averageScore
      popularity
      tags { name rank }
      studios(isMain: true) { nodes { id name } }
      staff(perPage: 15) { edges { role node { id name { full } } } }
      relations { edges { relationType node { id type } } }
    }
  }
}
"""


def fetch_page(page: int) -> dict:
    r = requests.post(API_URL, json={"query": QUERY, "variables": {"page": page}}, timeout=30)
    if r.status_code == 429:
        wait = int(r.headers.get("Retry-After", 60))
        print(f"  rate limited, sleeping {wait}s")
        time.sleep(wait + 1)
        return fetch_page(page)
    r.raise_for_status()
    return r.json()["data"]["Page"]


def main(pages: int) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    for page in range(1, pages + 1):
        data = fetch_page(page)
        path = f"{RAW_DIR}/page_{page:03d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data["media"], f, ensure_ascii=False)
        print(f"page {page}/{pages} -> {path}")
        if not data["pageInfo"]["hasNextPage"]:
            break
        time.sleep(DELAY)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
