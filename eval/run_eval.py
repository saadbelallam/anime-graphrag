"""python -m eval.run_eval [--failures]"""

import argparse
import json
import time
from collections import defaultdict

from graphrag.pipeline import ask

QUESTIONS = "eval/questions.json"
RESULTS = "eval/results.json"


def matches(answer: str, expected: list[str]) -> bool:
    answer = answer.lower()
    return any(e.lower() in answer for e in expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failures", action="store_true", help="print failed questions")
    args = parser.parse_args()

    with open(QUESTIONS, encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['question'][:70]}")
        try:
            r = ask(q["question"])
        except Exception as e:
            print(f"    error: {e}")
            results.append({**q, "mode": "error", "route_ok": False, "answer_ok": False, "answer": ""})
            continue

        route_ok = r.mode == q["expected_mode"]
        answer_ok = matches(r.answer, q["expected_contains"]) if q["expected_contains"] else None

        route_note = r.mode if route_ok else f"{r.mode} (expected {q['expected_mode']})"
        answer_note = "-" if answer_ok is None else ("ok" if answer_ok else "miss")
        print(f"    {route_note:<28} {answer_note}")

        results.append({**q, "mode": r.mode, "route_ok": route_ok, "answer_ok": answer_ok, "answer": r.answer})
        time.sleep(1)

    scored = [r for r in results if r["answer_ok"] is not None]
    by_mode = defaultdict(lambda: [0, 0])
    for r in scored:
        by_mode[r["expected_mode"]][1] += 1
        by_mode[r["expected_mode"]][0] += r["answer_ok"]

    print("\n" + "-" * 40)
    print(f"routing  {sum(r['route_ok'] for r in results)}/{len(results)}")
    print(f"answers  {sum(r['answer_ok'] for r in scored)}/{len(scored)}")
    for mode, (ok, total) in sorted(by_mode.items()):
        print(f"  {mode:<8} {ok}/{total}")

    if args.failures:
        for r in results:
            if r["answer_ok"] is False:
                print(f"\n[{r['expected_mode']}] {r['question']}")
                print(f"  expected  {r['expected_contains']}")
                print(f"  got       {r['answer'][:180]}")

    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
