"""Embed every title and create the Neo4j vector index."""

from sentence_transformers import SentenceTransformer

from graphrag.config import EMBEDDING_DIMS, EMBEDDING_MODEL, get_driver

MAX_SYNOPSIS = 2000


def document_text(row: dict) -> str:
    parts = [row["title"]]
    if row["genres"]:
        parts.append("Genres: " + ", ".join(row["genres"]))
    if row["tags"]:
        parts.append("Tags: " + ", ".join(row["tags"]))
    if row["synopsis"]:
        parts.append(row["synopsis"][:MAX_SYNOPSIS])
    return "passage: " + "\n".join(parts)


def main() -> None:
    model = SentenceTransformer(EMBEDDING_MODEL)
    driver = get_driver()

    with driver.session() as s:
        rows = s.run("""
            MATCH (a:Anime)
            OPTIONAL MATCH (a)-[:TAGGED]->(t:Tag)
            RETURN a.id AS id, a.title AS title, a.genres AS genres,
                   a.synopsis AS synopsis, collect(t.name) AS tags
        """).data()
        print(f"{len(rows)} titles to embed")

        vectors = model.encode(
            [document_text(r) for r in rows],
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        s.run("""
            UNWIND $batch AS row
            MATCH (a:Anime {id: row.id}) SET a.embedding = row.v
        """, batch=[{"id": r["id"], "v": v.tolist()} for r, v in zip(rows, vectors)])

        s.run(f"""
            CREATE VECTOR INDEX anime_synopsis IF NOT EXISTS
            FOR (a:Anime) ON (a.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIMS},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)
        print("vector index ok")

    driver.close()


if __name__ == "__main__":
    main()
