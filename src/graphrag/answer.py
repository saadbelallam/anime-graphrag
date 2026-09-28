from graphrag.llm import chat
from graphrag.retrievers import KEY_ROLES


def build_context(session, anime_ids: list[int], anchors=None) -> str:
    anchor_ids = [a["id"] for a in (anchors or [])]
    rows = session.run(
        """
        MATCH (c:Anime) WHERE c.id IN $ids
        OPTIONAL MATCH (c)-[:PRODUCED_BY]->(s:Studio)
        OPTIONAL MATCH (c)<-[w:WORKED_ON]-(p:Staff) WHERE w.role IN $roles
        OPTIONAL MATCH (c)<-[wc:WORKED_ON]-(shared:Staff)-[wa:WORKED_ON]->(anchor:Anime)
            WHERE anchor.id IN $anchor_ids
              AND wc.role IN $roles AND wa.role IN $roles
        RETURN c.title AS title, c.year AS year, c.genres AS genres,
               c.score AS score, left(c.synopsis, 400) AS synopsis,
               collect(DISTINCT s.name) AS studios,
               collect(DISTINCT p.name + ' (' + w.role + ')')[..8] AS staff,
               collect(DISTINCT shared.name + ' (' + wc.role
                       + '), aussi sur ' + anchor.title) AS links
        """,
        ids=anime_ids, anchor_ids=anchor_ids, roles=KEY_ROLES,
    ).data()

    blocks = []
    for row in rows:
        lines = [f"{row['title']} ({row['year'] or '?'})"]
        if row["studios"]:
            lines.append("  studio : " + ", ".join(row["studios"]))
        if row["staff"]:
            lines.append("  staff : " + ", ".join(row["staff"]))
        if row["links"]:
            lines.append("  liens : " + " ; ".join(row["links"]))
        if row["genres"]:
            lines.append("  genres : " + ", ".join(row["genres"]))
        if row["score"]:
            lines.append(f"  note : {row['score']}/100")
        if row["synopsis"]:
            lines.append("  synopsis : " + row["synopsis"].replace("\n", " "))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def rows_to_context(rows: list[dict]) -> str:
    return "\n".join(" | ".join(f"{k} : {v}" for k, v in row.items()) for row in rows)


ANSWER_SYSTEM = """Tu réponds à des questions sur les anime en t'appuyant
UNIQUEMENT sur le contexte fourni, qui contient des faits extraits d'une base
de données.

Règles :
- Ne cite aucun titre, nom ou fait absent du contexte.
- Quand tu recommandes une oeuvre, explique le lien concret (même studio,
  même réalisateur) si le contexte le mentionne.
- Si le contexte ne permet pas de répondre, dis-le plutôt que d'inventer.
- Réponds en français, de façon naturelle et concise."""


def generate(question: str, context: str) -> str:
    if not context.strip():
        return "Je n'ai rien trouvé dans la base qui permette de répondre."
    prompt = f"Question : {question}\n\nContexte :\n{context}"
    return chat(ANSWER_SYSTEM, prompt, temperature=0.3)
