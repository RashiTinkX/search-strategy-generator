#!/usr/bin/env python
"""
Self-test for the deterministic layer. No network, no LLM, no pytest.

Pins the rules that cross-model agreement depends on, so a later change to
canonical.py / candidates.py cannot quietly undo them.

    python eval/selftest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import canonical, pipeline, query_builder            # noqa: E402
from backend.candidates import (                                  # noqa: E402
    candidate_slate, matched_spans, mesh_only_blocks, span_groups)
from backend.mesh_index import get_index                          # noqa: E402

Q1 = ("Does optogenetic stimulation of the hippocampus improve memory "
      "consolidation in rodent models?")
Q2 = "single-cell RNA sequencing of microglia in Alzheimer's disease"
Q3 = "CRISPR-Cas9 off-target effects detection methods"

failures: list[str] = []


def check(name: str, cond, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}   {detail}")
        failures.append(name)


def main() -> int:
    ix = get_index()

    print("MeSH index")
    check("english label for D002493",
          ix.get("D002493").label == "Central Nervous System Diseases",
          repr(ix.get("D002493").label))
    check("no non-ASCII descriptor labels leak into headings",
          all(canonical.preferred_label(ix, d).isascii()
              for d in ("D002493", "D006624", "D009474")))

    print("\nexact-only resolution")
    check("real heading resolves", canonical.resolve_strict(ix, "Hippocampus") is not None)
    check("entry term resolves", canonical.resolve_strict(ix, "RNA Sequencing") is not None)
    check("hallucination is dropped, not fuzzy-matched",
          canonical.resolve_strict(ix, "Hippocampal Memory Consolidation Circuitry") is None)
    # "Hippocamp" is a substring of a real heading, so the old ladder's LIKE tier
    # would have auto-accepted "Hippocampus" for it; exact-only must not.
    check("fuzzy substring is NOT accepted",
          canonical.resolve_strict(ix, "Hippocamp") is None
          and any(d.label == "Hippocampus" for d in ix.search("Hippocamp", limit=8)))

    print("\nsubsumption pruning")
    hipp = [d.dui for d in (canonical.resolve_strict(ix, h) for h in
                            ("Hippocampus", "CA1 Region, Hippocampal",
                             "CA3 Region, Hippocampal", "Dentate Gyrus")) if d]
    kept = canonical.prune_subsumed(ix, hipp)
    check("descendants of Hippocampus are pruned", kept == [hipp[0]],
          f"kept={[canonical.preferred_label(ix, d) for d in kept]}")
    rodents = [d.dui for d in (canonical.resolve_strict(ix, h) for h in
                               ("Rodentia", "Mice", "Rats")) if d]
    check("Mice/Rats collapse into Rodentia",
          canonical.prune_subsumed(ix, rodents) == [rodents[0]])
    check("unrelated headings both survive",
          len(canonical.prune_subsumed(ix, [d.dui for d in (
              canonical.resolve_strict(ix, "Microglia"),
              canonical.resolve_strict(ix, "Alzheimer Disease"))])) == 2)

    print("\ncanonical ordering (same content, different proposal order -> same query)")
    a = [{"name": "pop", "slot": "population", "mesh": ["Rodentia"], "freetext": ["mouse"]},
         {"name": "int", "slot": "intervention", "mesh": ["Optogenetics"], "freetext": []}]
    b = [{"name": "intervention", "slot": "intervention", "mesh": ["Optogenetics"], "freetext": []},
         {"name": "animals", "slot": "population", "mesh": ["Rodentia", "Mice"], "freetext": ["Mouse"]}]
    qa = query_builder.compile_search(
        pipeline.finalize(canonical.canonicalize_blocks(a, ix)["blocks"], strict=True, ix=ix), {})
    qb = query_builder.compile_search(
        pipeline.finalize(canonical.canonicalize_blocks(b, ix)["blocks"], strict=True, ix=ix), {})
    check("two orderings compile to the same hash", qa["hash"] == qb["hash"],
          f"{qa['hash']} vs {qb['hash']}")

    print("\ncandidate slate")
    spans = {s["span"].lower(): s["label"] for s in matched_spans(Q1, ix)}
    check("longest match wins ('memory consolidation', not 'memory')",
          spans.get("memory consolidation") == "Memory Consolidation", str(spans))
    check("plural/singular ('rodent' -> Rodentia)", spans.get("rodent") == "Rodentia")
    labels2 = {c["label"] for c in candidate_slate(Q2, ix)["candidates"]}
    check("phrase pass finds Single-Cell Gene Expression Analysis",
          "Single-Cell Gene Expression Analysis" in labels2)
    check("phrase pass rejects near-miss prefixes (no hip-prosthesis noise)",
          not any("Arthroplasty" in x for x in
                  {c["label"] for c in candidate_slate(Q1, ix)["candidates"]}))
    labels3 = {c["label"] for c in candidate_slate(Q3, ix)["candidates"]}
    check("compound split reaches CRISPR-Cas Systems", "CRISPR-Cas Systems" in labels3)
    check("'off-target' stays unmatched (free-text territory)",
          any("off-target" in u.lower() for u in candidate_slate(Q3, ix)["unmatched"]))
    check("generic words never become blocks",
          not {"Disease", "Models, Theoretical"} & {b["mesh"][0] for b in mesh_only_blocks(Q3, ix)})

    print("\nspan grouping (the question decides what is ORed, not the model)")
    slate = candidate_slate(Q1, ix)
    groups = span_groups(slate)
    gid = {c["label"]: groups[c["id"]] for c in slate["candidates"]}
    check("a broader term shares its child's group",
          gid.get("Limbic System") == gid.get("Hippocampus"))
    check("unrelated facets stay in different groups",
          gid.get("Hippocampus") != gid.get("Optogenetics"))
    slate2 = candidate_slate(Q2, ix)
    g2 = {c["label"]: span_groups(slate2)[c["id"]] for c in slate2["candidates"]}
    check("overlapping spans share a group (RNA-seq ~ single-cell)",
          g2.get("Sequence Analysis, RNA") == g2.get("Single-Cell Gene Expression Analysis"))
    check("Microglia is its own group", g2.get("Microglia") != g2.get("Alzheimer Disease"))

    ids = {c["label"]: c["id"] for c in slate["candidates"]}
    sel_a = {"blocks": [{"slot": "intervention", "ids": [ids["Optogenetics"]]},
                        {"slot": "context", "ids": [ids["Hippocampus"]]},
                        {"slot": "outcome", "ids": [ids["Memory Consolidation"]]}]}
    sel_b = {"blocks": [{"slot": "intervention",
                         "ids": [ids["Optogenetics"], ids["Hippocampus"]]},
                        {"slot": "outcome", "ids": [ids["Memory Consolidation"]]}]}
    def compile_sel(sel):
        blocks, _ = pipeline._blocks_from_selection(sel, slate, group_by_span=True)
        return query_builder.compile_search(
            pipeline.finalize(canonical.canonicalize_blocks(blocks, ix)["blocks"],
                              strict=True, ix=ix), {})["hash"]
    check("same headings, different model grouping -> same query",
          compile_sel(sel_a) == compile_sel(sel_b),
          f"{compile_sel(sel_a)} vs {compile_sel(sel_b)}")

    print("\nmesh_only determinism")
    h = {query_builder.compile_search(
        pipeline.build(Q1, mode="mesh_only")["concepts"], {})["hash"] for _ in range(3)}
    check("three builds -> one hash", len(h) == 1, str(h))

    print("\nslot merging")
    blocks = [{"name": "m1", "slot": "method", "mesh": ["Sequence Analysis, RNA"], "freetext": []},
              {"name": "m2", "slot": "method", "mesh": ["Single-Cell Analysis"], "freetext": []},
              {"name": "c", "slot": "context", "mesh": ["Alzheimer Disease"], "freetext": []}]
    merged = canonical.canonicalize_blocks(blocks, ix, merge_slots=True)["blocks"]
    check("same-slot blocks merge into one OR block", len(merged) == 2,
          str([b["mesh"] for b in merged]))
    unlabelled = [{"name": "a", "slot": "other", "mesh": ["Microglia"], "freetext": []},
                  {"name": "b", "slot": "other", "mesh": ["Alzheimer Disease"], "freetext": []}]
    kept_other = canonical.canonicalize_blocks(unlabelled, ix, merge_slots=True)["blocks"]
    check("unslotted blocks are NOT merged (they stay ANDed)", len(kept_other) == 2,
          str([b["mesh"] for b in kept_other]))

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all checks passed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
