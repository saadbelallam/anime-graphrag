"""Sanity checks: LLM reachable, corpus loaded, vector search returns something sane."""

import httpx

from graphrag.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, get_driver
from graphrag.retrievers import vector_search

QUERIES = [
    ("fr", "anime de sport sur le volleyball"),
    ("en", "sports anime about volleyball"),
    ("fr", "mélancolique, deuil, perte d'un proche"),
    ("en", "melancholic, grief, loss of a loved one"),
]


def check_llm() -> None:
    print("== llm ==")
    print(f"url    {LLM_BASE_URL}")
    print(f"key    {'MISSING' if not LLM_API_KEY else LLM_API_KEY[:6] + '...'}")
    print(f"model  {LLM_MODEL}")
    try:
        r = httpx.get(f"{LLM_BASE_URL}/models",
                      headers={"Authorization": f"Bearer {LLM_API_KEY}"}, timeout=30)
        print(f"GET /models -> {r.status_code}")
        if r.status_code == 200:
            ids = [m.get("id") for m in r.json().get("data", [])]
            if LLM_MODEL not in ids:
                print(f"  warning: {LLM_MODEL} not in the {len(ids)} models listed")
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")


def check_corpus(session) -> None:
    print("\n== corpus ==")
    total = session.run("MATCH (a:Anime) RETURN count(a) AS n").single()["n"]
    embedded = session.run(
        "MATCH (a:Anime) WHERE a.embedding IS NOT NULL RETURN count(a) AS n"
    ).single()["n"]
    print(f"{total} titles, {embedded} embedded")


def check_vector(session) -> None:
    print("\n== vector search ==")
    for lang, q in QUERIES:
        print(f"\n[{lang}] {q}")
        for hit in vector_search(session, q, limit=5):
            print(f"   {hit['score']:.3f}  {hit['title']}")


def main() -> None:
    check_llm()
    driver = get_driver()
    try:
        with driver.session() as s:
            check_corpus(s)
            check_vector(s)
    except Exception as e:
        print(f"\nneo4j unreachable: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
