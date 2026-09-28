from graphrag.llm import chat_json

SYSTEM = """Tu es le routeur d'un système de question-réponse sur les anime.

Le système dispose de deux sources :
- un graphe Neo4j : anime, studios, staff avec leurs rôles, tags, et les
  relations entre oeuvres (suite, préquelle, spin-off)
- un index vectoriel sur les descriptions, pour les recherches par ambiance

Classe la question dans un de ces trois modes :
- "graph"  : fait ou relation précise (qui a réalisé, quel studio, combien,
             quelle suite...)
- "vector" : recherche par ambiance ou thème, sans oeuvre précise mentionnée
- "hybrid" : une oeuvre ou une personne précise EST mentionnée ET la question
             contient un critère flou (ressemblance, ambiance, "dans le même
             genre que")

Réponds UNIQUEMENT avec un JSON de cette forme, sans texte autour :
{"mode": "graph" | "vector" | "hybrid",
 "entities": ["titres ou noms mentionnés, tels quels"],
 "semantic_query": "la partie ambiance/thème, EN ANGLAIS, 5 à 10 mots-clés"}

semantic_query doit toujours être en anglais, même quand la question est en
français, car les descriptions du corpus sont en anglais. Laisse-la vide pour
le mode "graph"."""

MODES = ("graph", "vector", "hybrid")


def route(question: str) -> dict:
    try:
        decision = chat_json(SYSTEM, question)
    except ValueError:
        # small models sometimes wrap the JSON in prose; don't crash on it
        decision = {}
    if decision.get("mode") not in MODES:
        decision["mode"] = "vector"
    decision.setdefault("entities", [])
    decision.setdefault("semantic_query", "")
    return decision
