import re

from sentence_transformers import SentenceTransformer

from graphrag.config import EMBEDDING_MODEL
from graphrag.llm import chat, strip_fences

# roles that actually define a work; sharing a theme song singer is not a link
KEY_ROLES = [
    "Director",
    "Music",
    "Series Composition",
    "Character Design",
    "Original Creator",
    "Script",
]

_model = None


def embed_query(text: str) -> list[float]:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    # e5 expects "query: " here and "passage: " on the document side
    return _model.encode("query: " + text, normalize_embeddings=True).tolist()


LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')


def lucene_escape(text: str) -> str:
    # titles like "Haikyuu!!" or "Re:Zero" are Lucene syntax otherwise
    return LUCENE_SPECIAL.sub(r"\\\1", text)


def find_anchors(session, names: list[str]) -> list[dict]:
    anchors = []
    for name in names:
        rows = session.run(
            """
            CALL db.index.fulltext.queryNodes('anime_titles', $q)
            YIELD node, score
            RETURN node.id AS id, node.title AS title, score
            ORDER BY score DESC,
                     CASE node.format WHEN 'TV' THEN 0 ELSE 1 END,
                     node.popularity DESC
            LIMIT 1
            """,
            q=lucene_escape(name),
        ).data()
        if rows:
            anchors.append(rows[0])
    return anchors


SCHEMA = """Schéma du graphe Neo4j :
(:Anime {id, title, synonymsText, year, format, genres, score, popularity, synopsis})
(:Studio {id, name})
(:Staff {id, name})
(:Tag {name})

(Anime)-[:PRODUCED_BY]->(Studio)
(Staff)-[:WORKED_ON {role}]->(Anime)
    role : "Director", "Music", "Series Composition", "Character Design",
           "Original Creator", "Script"...
(Anime)-[:TAGGED {rank}]->(Tag)
(Anime)-[:RELATED_TO {type}]->(Anime)
    type : "SEQUEL", "PREQUEL", "SIDE_STORY", "SPIN_OFF", "ALTERNATIVE"

Un index fulltext 'anime_titles' couvre title et synonymsText. Les titres sont
en romaji, donc pour retrouver une oeuvre par son nom, préférer :
CALL db.index.fulltext.queryNodes('anime_titles', 'nom cherché') YIELD node, score
plutôt qu'une égalité sur title."""

CYPHER_SYSTEM = f"""Tu traduis une question en une requête Cypher en LECTURE SEULE.

{SCHEMA}

Règles :
- Lecture uniquement : MATCH, WHERE, RETURN, ORDER BY, LIMIT, CALL db.index...
- Toujours un LIMIT (25 maximum).
- Retourne des propriétés lisibles (title, name, role), jamais des noeuds entiers.
- Retourne TOUJOURS le titre de l'anime concerné en plus de l'information
  demandée. "RETURN s.name" ne suffit pas : le lecteur ne saura pas à quelle
  oeuvre le nom se rapporte. Écrire "RETURN a.title AS anime, s.name AS composer".
- Après un queryNodes servant à identifier une oeuvre précise, restreindre les
  ancrages : WITH node ORDER BY score DESC LIMIT 1.
- Réponds UNIQUEMENT avec la requête, sans backticks ni commentaire."""

WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD)\b", re.IGNORECASE
)


def _run_readonly(session, cypher: str) -> list[dict]:
    if WRITE_KEYWORDS.search(cypher):
        raise PermissionError("write query rejected")
    return session.run(cypher).data()


def text2cypher(session, question: str):
    cypher = strip_fences(chat(CYPHER_SYSTEM, question, temperature=0.0))
    try:
        return cypher, _run_readonly(session, cypher), None
    except Exception as first_error:
        # one repair attempt with the actual Neo4j error, then give up
        prompt = (
            f"Question : {question}\n\nTa requête :\n{cypher}\n\n"
            f"Neo4j renvoie :\n{first_error}\n\nCorrige-la."
        )
        cypher = strip_fences(chat(CYPHER_SYSTEM, prompt, temperature=0.0))
        try:
            return cypher, _run_readonly(session, cypher), None
        except Exception as second_error:
            return cypher, [], str(second_error)


def vector_search(session, query: str, limit=8) -> list[dict]:
    return session.run(
        """
        CALL db.index.vector.queryNodes('anime_synopsis', $limit, $v)
        YIELD node, score
        RETURN node.id AS id, node.title AS title, score
        """,
        v=embed_query(query), limit=limit,
    ).data()


def rank_candidates(session, query: str, candidate_ids: list[int], limit=8) -> list[dict]:
    # scored directly rather than through the index: a global top-k would
    # silently drop candidates that rank low corpus-wide
    return session.run(
        """
        MATCH (a:Anime) WHERE a.id IN $ids AND a.embedding IS NOT NULL
        RETURN a.id AS id, a.title AS title,
               vector.similarity.cosine(a.embedding, $v) AS score
        ORDER BY score DESC LIMIT $limit
        """,
        ids=candidate_ids, v=embed_query(query), limit=limit,
    ).data()


def neighborhood_ids(session, anime_id: int) -> list[int]:
    row = session.run(
        """
        MATCH (a:Anime {id: $id})
        OPTIONAL MATCH (a)-[:PRODUCED_BY]->(:Studio)<-[:PRODUCED_BY]-(x:Anime)
        OPTIONAL MATCH (a)<-[w1:WORKED_ON]-(:Staff)-[w2:WORKED_ON]->(y:Anime)
            WHERE w1.role IN $roles AND w2.role IN $roles
        OPTIONAL MATCH (a)-[:RELATED_TO*1..2]-(z:Anime)
        RETURN collect(DISTINCT x.id) + collect(DISTINCT y.id)
             + collect(DISTINCT z.id) AS ids
        """,
        id=anime_id, roles=KEY_ROLES,
    ).single()
    return [i for i in row["ids"] if i is not None and i != anime_id]


def hybrid_search(session, anchors: list[dict], query: str, limit=8) -> list[dict]:
    candidates = set()
    for anchor in anchors:
        candidates.update(neighborhood_ids(session, anchor["id"]))
    if not candidates:
        return []
    return rank_candidates(session, query or "similar anime", list(candidates), limit)
