"""python scripts/ask.py "Qui a composé la musique de Shingeki no Kyojin ?" """

import argparse

from graphrag.pipeline import ask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("-q", "--quiet", action="store_true", help="answer only")
    args = parser.parse_args()

    result = ask(args.question)

    if not args.quiet:
        print(f"[route]   {result.mode}  entities={result.entities}")
        if result.semantic_query:
            print(f"[query]   {result.semantic_query}")
        if result.anchors:
            print(f"[anchors] {[a['title'] for a in result.anchors]}")
        if result.cypher:
            print(f"[cypher]\n{result.cypher}")
        if result.fallback:
            print(f"[fallback] {result.fallback}")
        print()

    print(result.answer)


if __name__ == "__main__":
    main()
