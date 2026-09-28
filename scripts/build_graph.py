"""Load data/raw/ into Neo4j and create the fulltext index."""

import glob
import json
import re

from graphrag.config import RAW_DIR, get_driver

MIN_TAG_RANK = 40

CONSTRAINTS = [
    "FOR (a:Anime) REQUIRE a.id IS UNIQUE",
    "FOR (s:Studio) REQUIRE s.id IS UNIQUE",
    "FOR (p:Staff) REQUIRE p.id IS UNIQUE",
    "FOR (t:Tag) REQUIRE t.name IS UNIQUE",
]


def clean_description(html: str | None) -> str:
    if not html:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", html)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\(Source:.*?\)\s*$", "", text, flags=re.DOTALL)
    return text.strip()


def load_raw() -> dict:
    animes = {}
    for path in sorted(glob.glob(f"{RAW_DIR}/page_*.json")):
        with open(path, encoding="utf-8") as f:
            for media in json.load(f):
                animes[media["id"]] = media
    return animes


def extract(animes: dict):
    known = set(animes)
    nodes, studios, staff, tags, relations = [], [], [], [], []

    for m in animes.values():
        title = m["title"]["romaji"] or m["title"]["english"] or ""
        synonyms = [s for s in [m["title"]["english"], *(m["synonyms"] or [])] if s]
        nodes.append({
            "id": m["id"],
            "title": title,
            "synonymsText": " | ".join(synonyms),
            "year": m.get("seasonYear"),
            "format": m.get("format"),
            "genres": m.get("genres") or [],
            "score": m.get("averageScore"),
            "popularity": m.get("popularity"),
            "synopsis": clean_description(m.get("description")),
        })
        for s in m["studios"]["nodes"]:
            studios.append({"anime": m["id"], "studio": s["id"], "name": s["name"]})
        for e in m["staff"]["edges"]:
            # "Director (eps 1-12)" -> "Director"
            role = re.sub(r"\s*\(.*\)", "", e["role"] or "").strip()
            staff.append({
                "anime": m["id"],
                "staff": e["node"]["id"],
                "name": e["node"]["name"]["full"],
                "role": role,
            })
        for t in m["tags"] or []:
            if t["rank"] and t["rank"] >= MIN_TAG_RANK:
                tags.append({"anime": m["id"], "tag": t["name"], "rank": t["rank"]})
        for e in m["relations"]["edges"]:
            target = e["node"]
            if target["type"] == "ANIME" and target["id"] in known:
                relations.append({"src": m["id"], "dst": target["id"], "type": e["relationType"]})

    return nodes, studios, staff, tags, relations


def main() -> None:
    animes = load_raw()
    if not animes:
        raise SystemExit(f"nothing in {RAW_DIR}/, run scripts/collect.py first")
    print(f"{len(animes)} titles loaded")

    nodes, studios, staff, tags, relations = extract(animes)

    driver = get_driver()
    with driver.session() as s:
        for c in CONSTRAINTS:
            s.run(f"CREATE CONSTRAINT IF NOT EXISTS {c}")

        s.run("UNWIND $rows AS r MERGE (a:Anime {id: r.id}) SET a += r", rows=nodes)
        print(f"  {len(nodes):>6} Anime")

        s.run("""
            UNWIND $rows AS r
            MATCH (a:Anime {id: r.anime})
            MERGE (st:Studio {id: r.studio}) SET st.name = r.name
            MERGE (a)-[:PRODUCED_BY]->(st)
        """, rows=studios)
        print(f"  {len(studios):>6} PRODUCED_BY")

        s.run("""
            UNWIND $rows AS r
            MATCH (a:Anime {id: r.anime})
            MERGE (p:Staff {id: r.staff}) SET p.name = r.name
            MERGE (p)-[:WORKED_ON {role: r.role}]->(a)
        """, rows=staff)
        print(f"  {len(staff):>6} WORKED_ON")

        s.run("""
            UNWIND $rows AS r
            MATCH (a:Anime {id: r.anime})
            MERGE (t:Tag {name: r.tag})
            MERGE (a)-[:TAGGED {rank: r.rank}]->(t)
        """, rows=tags)
        print(f"  {len(tags):>6} TAGGED")

        s.run("""
            UNWIND $rows AS r
            MATCH (a:Anime {id: r.src}), (b:Anime {id: r.dst})
            MERGE (a)-[:RELATED_TO {type: r.type}]->(b)
        """, rows=relations)
        print(f"  {len(relations):>6} RELATED_TO")

        s.run("""
            CREATE FULLTEXT INDEX anime_titles IF NOT EXISTS
            FOR (a:Anime) ON EACH [a.title, a.synonymsText]
        """)
        print("  fulltext index ok")

    driver.close()


if __name__ == "__main__":
    main()
