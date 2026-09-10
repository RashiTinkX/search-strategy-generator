"""
Determinism measurement for the "closure" pipeline (facets.py self-consistency
voting + closure.py deterministic MeSH resolution), scored on the SAME FOUR
metrics as upstream sensein/search-strategy-generator's eval/ table, so a run
of this script is directly comparable to the numbers in that project's
README:

  within-model   same (mode, model, question), rerun N times -> largest
                 identical-hash group / N
  cross-model    same (mode, question), pooled across all --models -> largest
                 identical-hash group / total runs. Chance level = 1/n_models.
  heading Jaccard  mean pairwise Jaccard of the MeSH-heading SET used by each
                 run (ignoring order/formatting), across all runs for that
                 (mode, question) -- do different runs pick the same headings
                 even when the compiled query string isn't byte-identical?
  PMID Jaccard   (--retrieval only, hits live PubMed) mean pairwise Jaccard of
                 the PMIDs actually retrieved, one esearch per DISTINCT
                 compiled query in the group (not one per run) -- the metric
                 a reviewer actually cares about: do these strategies find
                 the same papers?

This script is NOT a port of test.py or of upstream's determinism_eval.py --
it's a from-scratch implementation of the same four definitions, so the
numbers are comparable without the code being copied.

Usage:
    .venv/Scripts/python.exe eval/eval_closure.py --models anthropic/claude-haiku-4.5 --runs 5
    .venv/Scripts/python.exe eval/eval_closure.py --models anthropic/claude-haiku-4.5,ollama/llama3.2:1b --runs 3
    .venv/Scripts/python.exe eval/eval_closure.py --modes closure --runs 5 --retrieval
"""
from __future__ import annotations

import argparse
import itertools
import json
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
from backend.pubmed import PubMed

DEFAULT_QUESTIONS = [
    "What is the effect of RNA sequencing on hippocampal neurons in Alzheimer Disease patients?",
    "Does metformin reduce cardiovascular mortality in patients with type 2 diabetes?",
    "Does mindfulness-based stress reduction reduce anxiety symptoms in breast cancer "
    "patients undergoing chemotherapy?",
]


def _headings(concepts: list[dict]) -> frozenset[str]:
    return frozenset(h.lower() for c in concepts for h in c.get("mesh", []))


def _run(question: str, mode: str, model: str, ix, k: int) -> dict | None:
    """One (mode, model, question) sample -> {hash, query, headings, model} or
    None on failure (reported, not raised, so one bad run doesn't kill a
    sweep)."""
    if mode == "llm":
        result = map_question(question, model=model)
        resolved = resolve_concepts(result, domains=[])
        concepts = facet_pipeline.concepts_for_compile(resolved["concepts"])
    elif mode == "closure":
        build = facet_pipeline.build(question, ix, model=model, k=k)
        concepts = facet_pipeline.concepts_for_compile(build["concepts"])
    else:
        raise ValueError(f"unknown mode {mode}")
    compiled = query_builder.compile_search(concepts, {})
    if not compiled["query"]:
        return None
    return {"hash": compiled["hash"], "query": compiled["query"],
            "headings": _headings(concepts), "model": model}


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _mean_pairwise_jaccard(sets: list[frozenset]) -> float | None:
    """Mean pairwise Jaccard across `sets`. A single set (e.g. every run
    produced the same query, so there's only one distinct query to compare)
    means perfect agreement by definition -> 1.0, not undefined; only a
    genuinely empty input (nothing succeeded) is None."""
    if not sets:
        return None
    if len(sets) == 1:
        return 1.0
    pairs = list(itertools.combinations(sets, 2))
    return round(sum(_jaccard(a, b) for a, b in pairs) / len(pairs), 3)


def _determinism(hashes: list[str]) -> float:
    if not hashes:
        return 0.0
    top = Counter(hashes).most_common(1)[0][1]
    return round(top / len(hashes), 3)


def _retrieval_jaccard(runs: list[dict], pm: PubMed, retmax: int = 500) -> dict:
    """PMID Jaccard over DISTINCT compiled queries in the group (fetching the
    same query's PMIDs once, not once per run that produced it)."""
    distinct_queries = sorted({r["query"] for r in runs})
    pmid_sets: list[frozenset] = []
    errors = []
    for q in distinct_queries:
        try:
            pmid_sets.append(frozenset(pm.ids(q, retmax=retmax)))
        except Exception as e:  # noqa: BLE001 - one bad query shouldn't kill retrieval scoring
            errors.append(str(e))
    jaccard = _mean_pairwise_jaccard(pmid_sets)
    return {"pmid_jaccard": jaccard, "distinct_queries": len(distinct_queries),
            "retrieval_errors": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ollama/llama3.2:1b",
                    help="comma-separated model ids; >1 enables cross-model scoring")
    ap.add_argument("--modes", default="llm,closure")
    ap.add_argument("--runs", type=int, default=5, help="runs per (mode, model, question)")
    ap.add_argument("--facet-k", type=int, default=3)
    ap.add_argument("--questions", default=None, help="path to a JSON list of question strings")
    ap.add_argument("--retrieval", action="store_true",
                    help="also fetch live PubMed PMIDs and score PMID Jaccard (network)")
    ap.add_argument("--out", default=str(ROOT / "data" / "closure_determinism.json"))
    args = ap.parse_args()

    questions = DEFAULT_QUESTIONS
    if args.questions:
        questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    ix = get_index()
    pm = PubMed() if args.retrieval else None

    report = {"models": models, "runs_per_cell": args.runs, "facet_k": args.facet_k,
             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": []}

    for question in questions:
        for mode in modes:
            all_runs: list[dict] = []          # every (model, run) sample, pooled
            per_model: dict[str, list[dict]] = {}
            for model in models:
                runs: list[dict] = []
                for i in range(args.runs):
                    try:
                        r = _run(question, mode, model, ix, args.facet_k)
                    except (OpenRouterError, Exception) as e:  # noqa: BLE001
                        print(f"  [{mode}/{model}] run {i+1}/{args.runs} failed: {e}",
                             file=sys.stderr)
                        r = None
                    if r:
                        runs.append(r)
                        all_runs.append(r)
                per_model[model] = runs
                within = _determinism([r["hash"] for r in runs]) if runs else 0.0
                print(f"{mode:8s} | {model:32s} | {question[:45]:45s} | "
                     f"within-model={within:.2f} (n={len(runs)}/{args.runs})")

            cross_model = (_determinism([r["hash"] for r in all_runs])
                          if len(models) > 1 and all_runs else None)
            heading_jaccard = _mean_pairwise_jaccard([r["headings"] for r in all_runs])
            retrieval = _retrieval_jaccard(all_runs, pm) if (args.retrieval and all_runs) else {}

            cell = {
                "mode": mode, "question": question,
                "within_model": {m: _determinism([r["hash"] for r in rs]) if rs else 0.0
                                 for m, rs in per_model.items()},
                "cross_model": cross_model,
                "heading_jaccard": heading_jaccard,
                **retrieval,
                "total_runs": len(all_runs), "total_attempted": len(models) * args.runs,
            }
            report["results"].append(cell)
            xm = f"{cross_model:.2f}" if cross_model is not None else "n/a (1 model)"
            hj = f"{heading_jaccard:.2f}" if heading_jaccard is not None else "n/a"
            pj = retrieval.get("pmid_jaccard")
            pj_s = f"{pj:.2f}" if pj is not None else "n/a"
            print(f"  -> cross-model={xm}  heading-jaccard={hj}  pmid-jaccard={pj_s}\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")

    print("\n--- summary (mean across questions, per mode) ---")
    for mode in modes:
        rows = [r for r in report["results"] if r["mode"] == mode]
        wm = [v for r in rows for v in r["within_model"].values()]
        xm = [r["cross_model"] for r in rows if r["cross_model"] is not None]
        hj = [r["heading_jaccard"] for r in rows if r["heading_jaccard"] is not None]
        pj = [r.get("pmid_jaccard") for r in rows if r.get("pmid_jaccard") is not None]
        print(f"{mode:8s}: within-model={_avg(wm)}  cross-model={_avg(xm)}  "
             f"heading-jaccard={_avg(hj)}  pmid-jaccard={_avg(pj)}")


def _avg(vals: list[float]) -> str:
    return f"{sum(vals)/len(vals):.3f}" if vals else "n/a"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, frozenset):
        return sorted(obj)
    return obj


if __name__ == "__main__":
    main()
