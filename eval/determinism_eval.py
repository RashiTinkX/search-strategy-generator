#!/usr/bin/env python
"""
Determinism / reproducibility evaluation for the deterministic PubMed search.

Measures four things, per strategy (mode) and per model:

  within-model     same model, same question, N runs -> how often the compiled
                   query is byte-identical (reproducibility for one lab)
  cross-model      different models, same question -> do they compile to the SAME
                   query (is the protocol portable between labs)
  heading Jaccard  cross-model overlap of the SELECTED MeSH heading sets — a
                   semantic view that does not punish formatting
  retrieval        (optional, --retrieval) cross-model overlap of the PMID sets
                   PubMed actually returns; the only metric a reviewer cares
                   about, since two different strings can retrieve one corpus

Modes: llm (free proposal, prompt v1 or v2), hybrid (LLM selects from a
deterministic MeSH candidate slate), mesh_only (no LLM).

IMPORTANT about caching: the earlier run of this experiment used a map cache
keyed by (model, question), which pins the LLM output and therefore makes
within-model determinism 1.00 BY CONSTRUCTION. Caching is off by default here;
--cache exists only for cheap re-runs of the deterministic layer.

Usage
    python eval/determinism_eval.py --runs 3 --modes hybrid,llm,mesh_only
    python eval/determinism_eval.py --models "openai/gpt-5.4-mini,google/gemini-3.5-flash"
    python eval/determinism_eval.py --prompt-versions v1,v2 --modes llm   # A/B the prompt
    python eval/determinism_eval.py --retrieval            # + live PubMed PMID overlap
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_env(path: Path = ROOT / ".env") -> None:
    """run.sh sources .env for the server; do the same for the CLI."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

from backend import pipeline, query_builder                      # noqa: E402
from backend.openrouter_client import OpenRouterError            # noqa: E402

DEFAULT_MODELS = [
    "anthropic/claude-opus-4.8",
    "google/gemini-3.5-flash",
    "deepseek/deepseek-v3.2",
    "openai/gpt-5.4-mini",
    "mistralai/ministral-8b-2512",
    "meta-llama/llama-3.1-8b-instruct",
    "microsoft/phi-4",
]

DEFAULT_QUESTIONS = [
    {"question": "Does optogenetic stimulation of the hippocampus improve memory "
                 "consolidation in rodent models?", "domains": ["neuroscience"]},
    {"question": "single-cell RNA sequencing of microglia in Alzheimer's disease",
     "domains": ["bioinformatics", "neuroscience"]},
    {"question": "CRISPR-Cas9 off-target effects detection methods",
     "domains": ["bioinformatics"]},
]

CACHE_DIR = ROOT / "data" / "eval_cache"


# ---------------------------------------------------------------- one run

def _cache_key(mode, model, prompt_version, question, domains) -> str:
    raw = json.dumps([mode, model, prompt_version, question, sorted(domains)], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def one_run(mode: str, model: str, prompt_version: str, item: dict, *,
            strict: bool, use_cache: bool) -> dict:
    """Build + compile once. Returns the run record (never raises)."""
    question, domains = item["question"], item.get("domains", [])
    t0 = time.monotonic()
    cache_file = CACHE_DIR / f"{_cache_key(mode, model, prompt_version, question, domains)}.json"
    try:
        if use_cache and cache_file.exists():
            build = json.loads(cache_file.read_text())
        else:
            build = pipeline.build(question, domains=domains, mode=mode,
                                   model=None if mode == "mesh_only" else model,
                                   prompt_version=prompt_version, strict=strict,
                                   fallback=True)
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(build))
        compiled = query_builder.compile_search(build["concepts"], {})
        headings = sorted({h for c in build["concepts"] for h in c["mesh"]})
        return {
            "ok": True,
            "latency_s": round(time.monotonic() - t0, 2),
            "query": compiled["query"],
            "hash": compiled["hash"],
            "n_concepts": compiled["n_concepts"],
            "n_terms": compiled["n_terms"],
            "n_chars": len(compiled["query"]),
            "headings": headings,
            "slots": sorted(c.get("slot", "other") for c in build["concepts"]),
            "dropped": len(build["dropped"]),
            "invalid_ids": len(build.get("invalid_ids") or []),
            "notes": build["notes"][:400],
        }
    except (OpenRouterError, ValueError, KeyError, TypeError) as e:
        return {"ok": False, "latency_s": round(time.monotonic() - t0, 2),
                "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- metrics

def determinism(values: list[str]) -> dict:
    """Modal share: 1.0 = every run identical, 1/n = all runs distinct."""
    if not values:
        return {"runs": 0, "distinct": 0, "score": 0.0}
    counts = Counter(values)
    modal = counts.most_common(1)[0][1]
    return {"runs": len(values), "distinct": len(counts),
            "score": round(modal / len(values), 4)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def mean_pairwise_jaccard(sets: list[set]) -> float:
    pairs = list(itertools.combinations(sets, 2))
    if not pairs:
        return 1.0
    return round(statistics.fmean(jaccard(a, b) for a, b in pairs), 4)


def cross_model(per_model: dict[str, dict]) -> dict:
    """
    Agreement between models on one question.

    `agreement` is the modal share of byte-identical queries (the strict view);
    `heading_jaccard` is the mean pairwise overlap of heading sets (the semantic
    view); `block_count_agreement` says whether they even agree on how many
    facets the question has.
    """
    hashes = [r["hash"] for r in per_model.values() if r.get("ok")]
    headings = [set(r["headings"]) for r in per_model.values() if r.get("ok")]
    nblocks = [r["n_concepts"] for r in per_model.values() if r.get("ok")]
    out = determinism(hashes)
    return {
        "n_models": len(hashes),
        "distinct_queries": out["distinct"],
        "agreement": out["score"],
        "heading_jaccard": mean_pairwise_jaccard(headings),
        "block_counts": dict(sorted(Counter(nblocks).items())),
        "block_count_agreement": determinism([str(n) for n in nblocks])["score"],
        "consensus_models": sorted(
            m for m, r in per_model.items()
            if r.get("ok") and r["hash"] == Counter(hashes).most_common(1)[0][0]
        ) if hashes else [],
    }


def retrieval_overlap(queries: dict[str, str], api_key: str, email: str,
                      cap: int = 20000) -> dict:
    """Cross-model overlap of the PMID sets PubMed returns (live)."""
    from backend.pubmed import PubMed
    pm = PubMed(api_key=api_key, email=email)
    sets: dict[str, set] = {}
    counts: dict[str, int] = {}
    for model, q in sorted(queries.items()):
        try:
            res = pm.search(q)
            counts[model] = res["count"]
            sets[model] = set(pm.pmids(q, retmax=min(cap, res["count"]))) if res["count"] else set()
        except Exception as e:                                    # network/NCBI hiccup
            counts[model] = -1
            sets[model] = set()
            print(f"    ! retrieval failed for {model}: {e}", flush=True)
    vals = [c for c in counts.values() if c >= 0]
    return {
        "counts": counts,
        "count_spread": (max(vals) - min(vals)) if vals else 0,
        "median_count": int(statistics.median(vals)) if vals else 0,
        "pmid_jaccard": mean_pairwise_jaccard([s for s in sets.values() if s]),
    }


# ---------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--modes", default="hybrid,llm,mesh_only")
    ap.add_argument("--prompt-versions", default="v2",
                    help="llm mode only; comma-separated (v1,v2) to A/B the prompt")
    ap.add_argument("--runs", type=int, default=3, help="runs per (mode, model, question)")
    ap.add_argument("--questions", default="", help="JSON file: [{question, domains}]")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-strict", action="store_true",
                    help="keep the model's free-text instead of MeSH-derived terms")
    ap.add_argument("--cache", action="store_true",
                    help="pin each (mode,model,question) build; makes within-model 1.00 by "
                         "construction — only for re-testing the deterministic layer")
    ap.add_argument("--retrieval", action="store_true",
                    help="also measure live PubMed PMID overlap across models")
    ap.add_argument("--out", default=str(ROOT / "data" / "determinism_v2.json"))
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    pvs = [p.strip() for p in args.prompt_versions.split(",") if p.strip()]
    questions = (json.loads(Path(args.questions).read_text())
                 if args.questions else DEFAULT_QUESTIONS)
    strict = not args.no_strict

    # mesh_only has no model and no prompt; run it once per question.
    tasks = []
    for mode in modes:
        if mode == "mesh_only":
            for qi, item in enumerate(questions):
                tasks.append((mode, "-", "-", qi, item, 0))
            continue
        for pv in (pvs if mode == "llm" else ["-"]):
            for model in models:
                for qi, item in enumerate(questions):
                    for r in range(args.runs):
                        tasks.append((mode, model, pv, qi, item, r))

    print(f"Modes: {modes}   models: {len(models)}   questions: {len(questions)}   "
          f"runs: {args.runs}   strict: {strict}   cache: {args.cache}")
    print(f"Dispatching {len(tasks)} builds across {args.workers} workers...", flush=True)

    results: dict[tuple, list[dict]] = {}
    done = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one_run, mode, model, pv, item, strict=strict,
                            use_cache=args.cache): (mode, model, pv, qi)
                for (mode, model, pv, qi, item, _r) in tasks}
        for fut, key in futs.items():
            rec = fut.result()
            results.setdefault(key, []).append(rec)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  ... {done}/{len(tasks)}", flush=True)
    wall = round(time.monotonic() - t0, 1)

    # ---- aggregate ----
    report: dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_seconds": wall,
                    "runs_per_cell": args.runs, "strict": strict, "cache": args.cache,
                    "models": models, "modes": modes, "prompt_versions": pvs,
                    "questions": [q["question"] for q in questions], "cells": [], "by_mode": {}}

    for mode in modes:
        for pv in (pvs if mode == "llm" else ["-"]):
            label = f"{mode}" + (f"/{pv}" if mode == "llm" else "")
            within_scores, cross, cells = [], [], []
            model_list = ["-"] if mode == "mesh_only" else models
            for qi, item in enumerate(questions):
                per_model: dict[str, dict] = {}
                for model in model_list:
                    runs = results.get((mode, model, pv, qi), [])
                    ok = [r for r in runs if r.get("ok")]
                    cell = {
                        "mode": label, "model": model, "question": item["question"],
                        "ok_runs": len(ok), "errors": [r["error"] for r in runs if not r.get("ok")],
                        "within_model": determinism([r["hash"] for r in ok]),
                        "n_concepts": sorted({r["n_concepts"] for r in ok}),
                        "n_terms": sorted({r["n_terms"] for r in ok}),
                        "n_chars": sorted({r["n_chars"] for r in ok}),
                        "headings": ok[0]["headings"] if ok else [],
                        "dropped": ok[0]["dropped"] if ok else 0,
                        "invalid_ids": sum(r.get("invalid_ids", 0) for r in ok),
                        "sample_query": ok[0]["query"] if ok else "",
                    }
                    cells.append(cell)
                    if ok:
                        within_scores.append(cell["within_model"]["score"])
                        per_model[model.split("/")[-1]] = ok[0]
                cm = cross_model(per_model)
                cm["question"] = item["question"]
                if args.retrieval and per_model:
                    cm["retrieval"] = retrieval_overlap(
                        {m: r["query"] for m, r in per_model.items()},
                        os.environ.get("NCBI_API_KEY", ""), os.environ.get("NCBI_EMAIL", ""))
                cross.append(cm)
            report["cells"].extend(cells)
            report["by_mode"][label] = {
                "within_model_mean": round(statistics.fmean(within_scores), 4) if within_scores else 0.0,
                "cross_model_mean": round(statistics.fmean(c["agreement"] for c in cross), 4),
                "heading_jaccard_mean": round(statistics.fmean(c["heading_jaccard"] for c in cross), 4),
                "block_count_agreement_mean": round(
                    statistics.fmean(c["block_count_agreement"] for c in cross), 4),
                "per_question": cross,
            }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print_report(report)
    print(f"\nReport -> {args.out}")
    return 0


def print_report(report: dict) -> None:
    print("\n" + "=" * 92)
    print(f"{'strategy':22s}{'within-model':>14s}{'cross-model':>13s}"
          f"{'heading Jaccard':>17s}{'block-count agr.':>18s}")
    print("-" * 92)
    for label, m in report["by_mode"].items():
        print(f"{label:22s}{m['within_model_mean']:>14.2f}{m['cross_model_mean']:>13.2f}"
              f"{m['heading_jaccard_mean']:>17.2f}{m['block_count_agreement_mean']:>18.2f}")
    print("=" * 92)
    print("within-model = same model reruns compile to the same query (reproducibility)")
    print("cross-model  = different models compile to the SAME query (portability)")
    print("heading Jaccard = semantic overlap of chosen MeSH headings across models")
    for label, m in report["by_mode"].items():
        print(f"\n{label}")
        for c in m["per_question"]:
            line = (f"  {c['question'][:52]:52s} agree={c['agreement']:.2f} "
                    f"jaccard={c['heading_jaccard']:.2f} blocks={c['block_counts']}")
            if "retrieval" in c:
                r = c["retrieval"]
                line += (f" pmid_jaccard={r['pmid_jaccard']:.2f} "
                         f"median_hits={r['median_count']} spread={r['count_spread']}")
            print(line)
    errs = [c for c in report["cells"] if c["errors"]]
    if errs:
        print("\nErrors:")
        for c in errs[:20]:
            print(f"  {c['mode']:12s} {c['model']:34s} {c['errors'][0][:90]}")


if __name__ == "__main__":
    raise SystemExit(main())
