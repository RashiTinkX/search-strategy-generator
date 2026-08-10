#!/usr/bin/env python
"""
Offline ablation: replay the ARCHIVED baseline LLM proposals through each stage of
the new deterministic layer.

`data/map_cache/*.json` holds the raw v1-prompt proposals from the original
11-model run (the LLM's own words, before any resolution). Replaying them lets us
measure exactly how much of the 0.09 cross-model agreement was the models
disagreeing and how much was our own pipeline, with the prompt and the models held
fixed — and it needs no API access.

Stages (each adds one deterministic rule):

  legacy       what the app did before: fuzzy heading resolution (substring match,
               auto-accept the first hit), no subsumption pruning, blocks kept in
               the model's order
  exact        drop headings that do not match a MeSH heading or entry term
               EXACTLY (no fuzzy fallback)
  +prune       also drop headings that a broader heading in the same block already
               explodes over
  +order       also sort blocks and terms canonically  = the new layer in full

Usage:  python eval/replay_baseline.py [--out data/replay_ablation.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import canonical, query_builder                     # noqa: E402
from backend.mesh_index import get_index                         # noqa: E402
from backend.pipeline import _blocks_from_concepts, finalize      # noqa: E402

# The cache file names are hashes made by the (now lost) original harness, so the
# question is recovered from the proposal's own vocabulary.
QUESTION_KEYS = [
    ("Does optogenetic stimulation of the hippocampus improve memory consolidation "
     "in rodent models?", ("optogenetic", "hippocamp")),
    ("single-cell RNA sequencing of microglia in Alzheimer's disease",
     ("microglia", "alzheimer")),
    ("CRISPR-Cas9 off-target effects detection methods", ("crispr", "cas9")),
]


def which_question(payload: dict) -> str | None:
    blob = json.dumps(payload).lower()
    scored = [(sum(blob.count(k) for k in keys), q) for q, keys in QUESTION_KEYS]
    best_score, best_q = max(scored)
    return best_q if best_score else None


def load_proposals(cache_dir: Path) -> dict[tuple[str, str], dict]:
    """(model, question) -> archived proposal. Later files win deterministically."""
    out: dict[tuple[str, str], dict] = {}
    for path in sorted(cache_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        model = payload.get("model") or "?"
        question = which_question(payload)
        if not question or not payload.get("concepts"):
            continue
        out[(model, question)] = payload
    return out


# ---------------------------------------------------------------- stages

def _resolve_fuzzy(ix, candidate: str) -> str:
    """The old behaviour: exact, else substring match, else the first suggestion."""
    hits = ix.exact(candidate)
    if hits:
        return hits[0].label
    options = ix.search(candidate, limit=8)
    return options[0].label if options else ""


def stage_blocks(payload: dict, ix, *, exact_only: bool, prune: bool, order: bool) -> list[dict]:
    blocks = _blocks_from_concepts(payload["concepts"])
    if exact_only:
        can = canonical.canonicalize_blocks(blocks, ix, prune=prune)
        out = can["blocks"]
        if not order:                       # keep the model's block order
            index = {}
            for i, b in enumerate(blocks):
                index.setdefault(b.get("name", ""), i)
            out = sorted(out, key=lambda b: index.get(b.get("name", ""), 99))
        return out
    resolved = []
    for b in blocks:
        labels, duis = [], []
        for cand in b.get("mesh", []):
            lab = _resolve_fuzzy(ix, cand)
            if not lab:
                continue
            hit = ix.exact(lab)
            if not hit:
                continue
            labels.append(lab)
            duis.append(hit[0].dui)
        if not labels and not b.get("freetext"):
            continue
        resolved.append({**b, "mesh": labels, "duis": duis})
    return resolved


def compile_stage(payload: dict, ix, **flags) -> dict:
    blocks = stage_blocks(payload, ix, **flags)
    concepts = finalize(blocks, strict=True, ix=ix)
    if not any(c["mesh"] or c["freetext"] for c in concepts):
        return {"ok": False}
    res = query_builder.compile_search(concepts, {})
    return {"ok": True, "hash": res["hash"], "n_concepts": res["n_concepts"],
            "n_terms": res["n_terms"], "n_chars": len(res["query"]),
            "headings": sorted({h for c in concepts for h in c["mesh"]}),
            "query": res["query"]}


# ---------------------------------------------------------------- metrics

def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def agreement(per_model: dict[str, dict]) -> dict:
    hashes = [r["hash"] for r in per_model.values() if r.get("ok")]
    heads = [set(r["headings"]) for r in per_model.values() if r.get("ok")]
    blocks = [r["n_concepts"] for r in per_model.values() if r.get("ok")]
    modal = Counter(hashes).most_common(1)[0][1] if hashes else 0
    pairs = list(itertools.combinations(heads, 2))
    return {
        "n_models": len(hashes),
        "distinct_queries": len(set(hashes)),
        "agreement": round(modal / len(hashes), 4) if hashes else 0.0,
        "heading_jaccard": round(statistics.fmean(jaccard(a, b) for a, b in pairs), 4) if pairs else 1.0,
        "block_counts": dict(sorted(Counter(blocks).items())),
        "median_terms": int(statistics.median(
            [r["n_terms"] for r in per_model.values() if r.get("ok")] or [0])),
        "median_chars": int(statistics.median(
            [r["n_chars"] for r in per_model.values() if r.get("ok")] or [0])),
    }


STAGES = [
    ("legacy (fuzzy, no prune, LLM order)", dict(exact_only=False, prune=False, order=False)),
    ("exact-only resolution",               dict(exact_only=True, prune=False, order=False)),
    ("+ subsumption pruning",               dict(exact_only=True, prune=True, order=False)),
    ("+ canonical ordering (new layer)",    dict(exact_only=True, prune=True, order=True)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "map_cache"))
    ap.add_argument("--out", default=str(ROOT / "data" / "replay_ablation.json"))
    args = ap.parse_args()

    ix = get_index()
    proposals = load_proposals(Path(args.cache_dir))
    models = sorted({m for m, _ in proposals})
    questions = [q for q, _ in QUESTION_KEYS]
    print(f"Replaying {len(proposals)} archived proposals from {len(models)} models "
          f"(prompt v1, temperature 0) through {len(STAGES)} stages\n")

    report = {"models": models, "questions": questions, "stages": {}}
    for label, flags in STAGES:
        per_question = []
        for q in questions:
            per_model = {}
            for m in models:
                payload = proposals.get((m, q))
                if not payload:
                    continue
                try:
                    per_model[m.split("/")[-1]] = compile_stage(payload, ix, **flags)
                except (ValueError, KeyError) as e:
                    per_model[m.split("/")[-1]] = {"ok": False, "error": str(e)}
            a = agreement(per_model)
            a["question"] = q
            per_question.append(a)
        report["stages"][label] = {
            "cross_model_mean": round(statistics.fmean(a["agreement"] for a in per_question), 4),
            "heading_jaccard_mean": round(statistics.fmean(a["heading_jaccard"] for a in per_question), 4),
            "per_question": per_question,
        }

    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"{'stage':38s}{'cross-model':>13s}{'heading Jaccard':>17s}")
    print("-" * 68)
    for label, s in report["stages"].items():
        print(f"{label:38s}{s['cross_model_mean']:>13.2f}{s['heading_jaccard_mean']:>17.2f}")
    print("-" * 68)
    for label, s in report["stages"].items():
        print(f"\n{label}")
        for a in s["per_question"]:
            print(f"  {a['question'][:48]:48s} agree={a['agreement']:.2f} "
                  f"jaccard={a['heading_jaccard']:.2f} distinct={a['distinct_queries']}/"
                  f"{a['n_models']} blocks={a['block_counts']} "
                  f"terms~{a['median_terms']} chars~{a['median_chars']}")
    print(f"\nReport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
