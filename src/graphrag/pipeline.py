from dataclasses import dataclass, field

from graphrag.answer import build_context, generate, rows_to_context
from graphrag.config import get_driver
from graphrag.retrievers import find_anchors, hybrid_search, text2cypher, vector_search
from graphrag.router import route


@dataclass
class Result:
    question: str
    mode: str
    answer: str = ""
    entities: list[str] = field(default_factory=list)
    semantic_query: str = ""
    anchors: list[dict] = field(default_factory=list)
    cypher: str | None = None
    fallback: str | None = None


def _graph(session, result: Result) -> str:
    cypher, rows, error = text2cypher(session, result.question)
    result.cypher = cypher
    if error:
        result.fallback = error
        return _vector(session, result)
    return rows_to_context(rows)


def _vector(session, result: Result) -> str:
    hits = vector_search(session, result.semantic_query or result.question)
    return build_context(session, [h["id"] for h in hits])


def _hybrid(session, result: Result) -> str:
    result.anchors = find_anchors(session, result.entities)
    if not result.anchors:
        result.fallback = "no entity resolved"
        return _vector(session, result)
    hits = hybrid_search(session, result.anchors, result.semantic_query)
    return build_context(session, [h["id"] for h in hits], result.anchors)


HANDLERS = {"graph": _graph, "vector": _vector, "hybrid": _hybrid}


def ask(question: str) -> Result:
    decision = route(question)
    result = Result(
        question=question,
        mode=decision["mode"],
        entities=decision["entities"],
        semantic_query=decision["semantic_query"],
    )

    driver = get_driver()
    try:
        with driver.session() as session:
            context = HANDLERS[result.mode](session, result)
    finally:
        driver.close()

    result.answer = generate(question, context)
    return result
