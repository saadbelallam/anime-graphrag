# 🕸️ anime-graphrag

> **Ask questions about anime in plain French. Get answers grounded in a knowledge graph, not in a language model's memory.**

A GraphRAG built on 1,500 anime from AniList. A Neo4j knowledge graph holds studios, staff, roles and the links between works; a vector index holds the descriptions. An LLM router decides, question by question, whether to query the graph, search the vectors, or combine both.

The point of the project is to answer one question: **what does a graph bring to a RAG, and on which kinds of questions?**

---

## The problem with plain RAG

A classic RAG chunks documents, embeds them, and retrieves the top-k nearest chunks. It works when the answer is written somewhere in a single passage. It breaks on two kinds of questions:

- **Relational questions.** *"Something by the director of Death Note"* requires knowing who directed Death Note, then what else they directed. No chunk contains that chain.
- **Questions with a named entity and a fuzzy criterion.** *"Something as moving as Violet Evergarden, from the same studio"* mixes a hard constraint (same studio) with a soft one (moving). Embedding the whole sentence pollutes the vector with the title, and the top result is... Violet Evergarden.

A vector index always returns something, even when nothing relevant exists. A graph query returns an empty set when the fact does not exist. **The graph knows how to say no.**

---

## 💡 How it works

```
                          question (French)
                                │
                                ▼
                    ┌───────────────────────┐
                    │      LLM router       │
                    │  mode + entities +    │
                    │  semantic query (EN)  │
                    └───────────┬───────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────────┐
   │    graph    │      │   vector    │      │     hybrid      │
   │             │      │             │      │                 │
   │ text2cypher │      │  cosine on  │      │ graph filters   │
   │ read-only,  │      │ descriptions│      │ the candidates, │
   │ 1 repair,   │      │             │      │ vector ranks    │
   │ fallback    │      │             │      │ them            │
   └──────┬──────┘      └──────┬──────┘      └────────┬────────┘
          │                    │                      │
          └────────────────────┼──────────────────────┘
                               ▼
                    ┌───────────────────────┐
                    │   context = facts     │
                    │   serialized from     │
                    │   the graph           │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │      generation       │
                    │  cannot cite anything │
                    │  outside the context  │
                    └───────────────────────┘
```

### The three strategies

| Mode | When | What happens |
|------|------|--------------|
| **graph** | Precise fact or relation: *who directed, which studio, how many, what sequel* | The LLM writes a Cypher query, Neo4j answers. Facts cannot be hallucinated. |
| **vector** | Mood or theme, no named work: *a melancholic anime about grief* | Cosine similarity between the query and the descriptions. |
| **hybrid** | A named work **and** a fuzzy criterion: *as moving as X, from the same studio* | The graph pulls X's neighborhood (same studio, shared key staff, sequels). The vector ranks only those candidates. |

### The router

A single LLM call returns a JSON: the mode, the entities mentioned, and the *semantic part* of the question rewritten in English. That split is what makes the hybrid mode work: only the semantic part gets embedded, the entity goes to the graph.

The semantic query is in English because the corpus is. The same query in French returns noise; in English it returns the right titles. Translating at the router costs nothing and fixes it.

---

## The graph

```
(:Anime {id, title, synonymsText, year, format, genres, score, popularity, synopsis, embedding})
(:Studio {id, name})
(:Staff {id, name})
(:Tag {name})

(Anime)-[:PRODUCED_BY]->(Studio)
(Staff)-[:WORKED_ON {role}]->(Anime)
(Anime)-[:TAGGED {rank}]->(Tag)
(Anime)-[:RELATED_TO {type}]->(Anime)      SEQUEL, PREQUEL, SIDE_STORY, SPIN_OFF...
```

Two indexes, both native to Neo4j 5:

- a **fulltext index** on titles and synonyms, to resolve a name mentioned in a question into a node
- a **vector index** on the embeddings, so a query can start with a similarity search and continue with a graph traversal without leaving the database

### Design choices worth knowing

**The role lives on the edge, not on the node.** The same person can direct one series and design characters on another. Putting the role on the `Staff` node would force duplicates and make *"who directed X"* impossible to ask.

**Not every shared credit is a link.** The hybrid neighborhood only follows `Director`, `Music`, `Series Composition`, `Character Design`, `Original Creator` and `Script`. Without that filter, two works end up linked because the same singer performed both opening themes.

**Titles resolve to TV series first.** The fulltext index would happily return *Violet Evergarden Movie* for *Violet Evergarden*. At equal score the resolver prefers TV format and higher popularity.

**Embeddings are built from title + genres + tags + synopsis.** A synopsis tells the plot, not the mood. The word *melancholic* is in no synopsis; it is in the tags.

**The generator only sees facts.** Retrieved nodes are serialized as attribute lists, with their explicit links to the anchor work. The prompt forbids citing anything absent from that context and asks the model to say so when it cannot answer.

---

## Getting started

### Prerequisites

- Python 3.10+
- Docker
- An API key for any OpenAI-compatible LLM endpoint (Mistral, OpenAI, a local vLLM, an internal gateway)

### Install

```bash
git clone https://github.com/<you>/anime-graphrag.git
cd anime-graphrag
pip install -e .
cp .env.example .env         # set LLM_API_KEY, adjust LLM_BASE_URL / LLM_MODEL if needed
docker compose up -d         # Neo4j on :7687, web UI on :7474 (neo4j / graphrag123)
```

### Build the database

```bash
python scripts/collect.py        # AniList -> data/raw/   (~2 min, rate-limited)
python scripts/build_graph.py    # data/raw/ -> Neo4j     (nodes, edges, fulltext index)
python scripts/embed.py          # embeddings + vector index (~2 min on CPU)
```

or simply `make build`. The embedding model runs locally and is downloaded on first use (~450 MB).

### Ask

```bash
python scripts/ask.py "Quel studio a produit Fullmetal Alchemist: Brotherhood ?"
python scripts/ask.py "Un anime mélancolique sur le deuil"
python scripts/ask.py "J'ai adoré Violet Evergarden, quelque chose d'aussi émouvant du même studio ?"
```

Without `--quiet`, the CLI prints the routing decision, the generated Cypher and the resolved anchors. When an answer is wrong, that is where you look first. The Cypher below is one possible output; it varies with the model.

```
$ python scripts/ask.py "Qui a composé la musique de Shingeki no Kyojin ?"
[route]   graph  entities=['Shingeki no Kyojin']
[cypher]
CALL db.index.fulltext.queryNodes('anime_titles', 'Shingeki no Kyojin') YIELD node, score
WITH node ORDER BY score DESC LIMIT 1
MATCH (s:Staff)-[:WORKED_ON {role: 'Music'}]->(node)
RETURN node.title AS anime, s.name AS composer

La musique de Shingeki no Kyojin a été composée par Hiroyuki Sawano.
```

---

## Evaluation

`eval/questions.json` holds 24 labelled questions split across the three modes. Each one carries the expected mode and a list of acceptable titles.

```bash
python -m eval.run_eval --failures
```

Two things are scored separately, and that separation is the whole point: **routing** (did the router pick the right strategy?) and **answers** (does the answer mention an acceptable title?). They fail for different reasons. A relational question sent down the vector path produces a fluent, plausible, wrong answer, and nothing flags it. Measuring them together would hide that.

`scripts/diagnose.py` runs the checks that matter before blaming the code: LLM reachable, corpus loaded, and raw vector search on the same query in French and in English.

---

## 📁 Project structure

```
anime-graphrag/
├── src/graphrag/
│   ├── config.py          # .env, Neo4j driver
│   ├── llm.py             # OpenAI-compatible client
│   ├── router.py          # mode + entities + semantic query
│   ├── retrievers.py      # text2cypher, vector, hybrid
│   ├── answer.py          # context serialization, generation
│   └── pipeline.py        # ask() ties it together
├── scripts/
│   ├── collect.py         # AniList -> data/raw/
│   ├── build_graph.py     # data/raw/ -> Neo4j
│   ├── embed.py           # embeddings + vector index
│   ├── ask.py             # CLI
│   └── diagnose.py        # sanity checks
├── eval/
│   ├── questions.json     # 24 labelled questions
│   └── run_eval.py
├── docker-compose.yml     # Neo4j 5
├── Makefile
└── .env.example
```

---

## 🛠️ Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Graph + vectors | [Neo4j 5](https://neo4j.com) | Native vector index: one store for the graph and the embeddings |
| Embeddings | [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | Local, free, 384 dims, good enough for a 1,500-title corpus |
| LLM | Any OpenAI-compatible endpoint | Router, Cypher generation, answer. Only the URL and model name change between providers |
| Data | [AniList GraphQL API](https://anilist.gitbook.io/anilist-apiv2-docs/) | Relations come pre-structured: no entity extraction step needed |

---

## Where to go next

- **HyDE for mood queries.** Comparing keywords against synopses is asymmetric. Generating a hypothetical synopsis from the question and comparing synopsis-to-synopsis should help on the questions the vector path still misses.
- **Characters and voice actors.** Not in the graph yet. *"An anime with the voice actress of X"* is the natural extension.
- **Faithfulness scoring.** The current eval checks that an acceptable title is present. A finer measure would split the answer into claims and verify each one against the retrieved context.
- **Parallel retrieval.** A routing error is silent. Running the three paths and merging with a reranker would trade cost for robustness.

---

## 📄 License

MIT. See [LICENSE](LICENSE).
