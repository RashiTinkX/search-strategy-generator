"""
Determinism measurement for the "closure" pipeline (facets.py self-consistency
voting + closure.py deterministic MeSH resolution), scored the same way
test.py already scores the legacy "llm" pipeline: run map -> compile N times
per (mode, model, question) and report

    query_hash determinism = (size of the largest identical-hash group) / N

so the numbers in this file's output are directly comparable to test.py's
and to the upstream sensein/search-strategy-generator eval/ table (both use
the same "does a rerun reproduce byte-identical compiled query" definition).
This script is NOT a port of test.py or of upstream's determinism_eval.py --
it is much shorter because it deliberately reuses query_builder.py directly
rather than re-deriving determinism_eval.py's report/plotting machinery,
which test.py in this repo already provides for the legacy pipeline.

Usage:
    .venv/Scripts/python.exe eval/eval_closure.py
    .venv/Scripts/python.exe eval/eval_closure.py --model ollama/llama3.2:1b --runs 5
    .venv/Scripts/python.exe eval/eval_closure.py --modes llm,closure --runs 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from backend import facet_pipeline, query_builder
from backend.app import resolve_concepts
from backend.mesh_index import get_index
from backend.openrouter_client import OpenRouterError, map_question

DEFAULT_QUESTIONS = [
    "What is the effect of RNA sequencing on hippocampal neurons in Alzheimer Disease patients?",
    "Does metformin reduce cardiovascular mortality in patients with type 2 diabetes?",
]


def _compile_hash(concepts: list[dict]) -> str | None:
    compiled = query_builder.compile_search(concepts, {})
    if not compiled["query"]:
        return None
    return compiled["hash"]


def run_llm_once(question: str, model: str) -> str | None:
    result = map_question(question, model=model)
    resolved = resolve_concepts(result, domains=[])
    concepts = facet_pipeline.concepts_for_compile(resolved["concepts"])
    return _compile_hash(concepts)


def run_closure_once(question: str, model: str, ix, k: int) -> str | None:
    build = facet_pipeline.build(question, ix, model=model, k=k)
    concepts = facet_pipeline.concepts_for_compile(build["concepts"])
    return _compile_hash(concepts)


def score(hashes: list[str | None]) -> dict:
    valid = [h for h in hashes if h]
    if not valid:
        return {"determinism": 0.0, "distinct": 0, "n": len(hashes), "failures": len(hashes)}
    counts = Counter(valid)
    top = counts.most_common(1)[0][1]
    return {"determinism": round(top / len(hashes), 3), "distinct": len(counts),
            "n": len(hashes), "failures": len(hashes) - len(valid)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ollama/llama3.2:1b")
    ap.add_argument("--modes", default="llm,closure")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--facet-k", type=int, default=3)
    ap.add_argument("--questions", default=None, help="path to a JSON list of question strings")
    ap.add_argument("--out", default=str(ROOT / "data" / "closure_determinism.json"))
    args = ap.parse_args()

    questions = DEFAULT_QUESTIONS
    if args.questions:
        questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    modes = args.modes.split(",")
    ix = get_index()

    report = {"model": args.model, "runs": args.runs, "facet_k": args.facet_k,
             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": []}

    for question in questions:
        for mode in modes:
            hashes: list[str | None] = []
            for i in range(args.runs):
                try:
                    if mode == "llm":
                        h = run_llm_once(question, args.model)
                    elif mode == "closure":
                        h = run_closure_once(question, args.model, ix, args.facet_k)
                    else:
                        raise ValueError(f"unknown mode {mode}")
                except (OpenRouterError, Exception) as e:  # noqa: BLE001 - report, don't crash the sweep
                    print(f"  [{mode}] run {i+1}/{args.runs} failed: {e}", file=sys.stderr)
                    h = None
                hashes.append(h)
            s = score(hashes)
            print(f"{mode:8s} | {question[:60]:60s} | determinism={s['determinism']:.2f} "
                 f"distinct={s['distinct']} failures={s['failures']}")
            report["results"].append({"mode": mode, "question": question, **s})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")

    by_mode: dict[str, list[float]] = {}
    for r in report["results"]:
        by_mode.setdefault(r["mode"], []).append(r["determinism"])
    print("\n--- summary (mean determinism across questions) ---")
    for mode, vals in by_mode.items():
        print(f"{mode:8s}: {sum(vals)/len(vals):.3f}  (n={len(vals)} questions)")


if __name__ == "__main__":
    main()
