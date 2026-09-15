"""
The question -> concept-blocks pipeline, in four modes.

One entry point shared by the web app and the evaluation harness, so what the
harness measures is exactly what the app does.

  llm        The LLM proposes blocks and headings freely (prompt v2 by default).
             Highest ceiling on judgement, widest output space.
  hybrid     A deterministic MeSH lookup over the question produces a numbered
             candidate slate; the LLM only SELECTS ids from it. A heading that is
             not in the slate cannot enter the query, so the model's whole output
             space is a function of (question, MeSH index) — this is what lifts
             cross-model agreement.
  closure    The LLM never sees MeSH at all. It only marks WHERE each PICO-style
             concept sits in the question text (a lower-entropy task than
             selecting from a slate), sampled k times and kept only where a
             majority of samples agree (self-consistency voting -- facets.py).
             Vocabulary is then resolved afterward as a pure function of
             (span, MeSH index) -- closure.py -- so the model's output space
             never includes a MeSH heading at all. Measured against this
             project's own eval/ definitions (within-model, cross-model,
             heading Jaccard, PMID Jaccard): 0.97 / 0.97 / 1.00 / 0.84 with
             Claude Haiku 4.5 + Sonnet 5 on 6 questions x 3 runs, vs hybrid's
             0.90 / 0.76 / 0.95 / 0.68 above. See facet_pipeline.py.
  mesh_only  No LLM at all: every maximal MeSH match in the question becomes a
             block. Trivially model-independent; the baseline and the fallback.

Whatever the mode, the LLM output passes through canonical.canonicalize_blocks
(exact-only resolution, subsumption pruning, canonical ordering) before it can
reach the query builder.
"""
from __future__ import annotations

from . import canonical, facet_pipeline
from .candidates import candidate_slate, group_closure, mesh_only_blocks, span_groups
from .domain_vocab import get_vocab
from .mesh_index import get_index
from .openrouter_client import (
    OpenRouterError,
    map_question,
    map_question_async,
    select_candidates,
    select_candidates_async,
)

MODES = ("llm", "hybrid", "closure", "mesh_only")


def _check_mode(mode: str) -> str:
    mode = (mode or "llm").strip().lower()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return mode


# ---------------------------------------------------------------- shaping

def _blocks_from_concepts(concepts: list[dict]) -> list[dict]:
    """LLM proposal (free-form) -> canonicaliser input."""
    return [{
        "name": c.get("name", ""),
        "slot": c.get("slot", "") or c.get("name", ""),
        "mesh": c.get("mesh_candidates", []) or c.get("mesh", []),
        "freetext": c.get("freetext", []),
        "explode": True,
        "rationale": c.get("rationale", ""),
    } for c in concepts]


def _blocks_from_selection(selection: dict, slate: dict, *,
                           group_by_span: bool = True,
                           closure: bool = True) -> tuple[list[dict], list[int]]:
    """
    Hybrid selection (candidate ids) -> canonicaliser input.

    Ids outside the slate are dropped: the model is not allowed to smuggle in
    vocabulary, and an out-of-range id is the one failure mode this design has.

    group_by_span: the model chooses WHICH candidates; the question decides how
    they group into ANDed blocks (candidates.span_groups). Two models that select
    the same headings then compile to the same query even if one of them would
    have ORed facets the other ANDed. The model's slot label, block name and
    free-text follow its ids into the group they land in.
    """
    by_id = {c["id"]: c for c in slate.get("candidates", [])}
    bad_ids: list[int] = []
    used: set[int] = set()
    groups = span_groups(slate) if group_by_span else {}

    grouped: dict[object, dict] = {}
    freetext_only: list[dict] = []
    for pos, b in enumerate(selection.get("blocks", [])):
        ids = []
        for i in b.get("ids", []):
            if i not in by_id:
                bad_ids.append(i)
                continue
            if i in used:            # an id may only anchor one block
                continue
            used.add(i)
            ids.append(i)
        if not ids:
            if b.get("freetext"):
                freetext_only.append({
                    "name": b.get("name", ""), "slot": b.get("slot", ""),
                    "mesh": [], "freetext": b.get("freetext", []), "explode": True,
                })
            continue

        for i in sorted(ids):
            key = groups.get(i, i) if group_by_span else pos
            g = grouped.setdefault(key, {"name": b.get("name", ""), "slot": b.get("slot", ""),
                                         "mesh": [], "freetext": [], "explode": True,
                                         "ids": []})
            g["mesh"].append(by_id[i]["label"])
            g["ids"].append(i)
            for t in b.get("freetext", []):
                if t not in g["freetext"]:
                    g["freetext"].append(t)

    if group_by_span and closure:
        # Replace the model's pick inside each facet with the facet's canonical
        # vocabulary; the model still decides which facets exist.
        canon = group_closure(slate, sorted(used))
        for key, g in grouped.items():
            if key in canon:
                g["mesh"] = [m["label"] for m in canon[key]]
        # A block with no MeSH id at all is the escape hatch for wording MeSH
        # lacks ("off-target"). Keeping the model's own list there was the last
        # prose channel into the query, and the widest: for one question the seven
        # models wrote "off target", "off-targets", "off-target activity",
        # "detection", "bioinformatics", "machine learning", "method*" — several of
        # which are over-restrictive as an ANDed block. So the model's judgement is
        # kept (does this question need a non-MeSH facet at all?) and the wording
        # is taken from the question itself: the phrases the MeSH lookup could not
        # resolve, as ONE block. Extend it by hand in review if the review needs
        # more jargon than the question contains.
        unmatched = canonical.sort_terms(slate.get("unmatched", []))
        freetext_only = ([{"name": "question terms not in MeSH", "slot": "other",
                           "mesh": [], "freetext": unmatched, "explode": True,
                           "derived": True}]
                         if (freetext_only and unmatched) else [])
    blocks = [grouped[k] for k in sorted(grouped, key=str)] + freetext_only
    return blocks, bad_ids


def _blocks_from_closure(concepts: list[dict]) -> list[dict]:
    """
    closure mode (facet_pipeline.build/build_async) -> canonicaliser input.

    facet_pipeline's concepts already carry resolved MeSH rows (each with its
    own single, exactly-matched descriptor -- see closure.py), so this is a
    reshape, not a resolution step: pull the label out of each matched row and
    drop the ones the frontend/UI-review shape carries that canonicalize_blocks
    doesn't need (pico_role, unmatched, dropped_subsumed). Re-running these
    through canonicalize_blocks below is redundant with what closure.py already
    did (both do exact-match resolution + subsumption pruning) but harmless --
    it keeps `closure` mode on the exact same _assemble()/finalize() path as
    the other three modes instead of a special-cased one.
    """
    out = []
    for c in concepts:
        mesh_labels = [m["options"][0]["label"] for m in c.get("mesh", [])
                       if m.get("matched") and m.get("options")]
        out.append({
            "name": c.get("name", ""),
            "slot": c.get("pico_role", "") or c.get("name", ""),
            "mesh": mesh_labels,
            "freetext": c.get("freetext", []),
            "explode": bool(c.get("explode", True)),
            "rationale": c.get("rationale", ""),
        })
    return out


def _vocab_terms(block: dict, domains: list[str] | None, *,
                 use_model_text: bool = False) -> list[str]:
    """
    Domain-vocabulary synonyms that apply to a block (deterministic, file-driven).

    Kept separate from the model's free-text so strict mode can drop the model's
    prose while keeping this — it is as reproducible as the MeSH index is.

    use_model_text=False (strict mode) matches only against the block's MeSH
    headings. Matching against the model's block NAME and free-text was a real
    determinism leak: two models that selected identical headings still compiled
    different queries (153 vs 145 terms) because their differing prose fired
    different vocabulary clusters.
    """
    parts = list(block.get("mesh", []))
    if use_model_text:
        parts += [block.get("name", "")] + list(block.get("freetext", []))
    hay = " ".join(parts).lower()
    out: list[str] = []
    for cl in get_vocab().clusters(domains or None):
        if cl.concept.lower() in hay or any(s.lower() in hay for s in cl.synonyms):
            out.extend(cl.synonyms)
    return canonical.sort_terms(out)


def finalize(blocks: list[dict], *, strict: bool = True, ix=None) -> list[dict]:
    """
    Blocks -> compile-shape concepts for query_builder.

    strict=True derives every free-text term for a block WITH MeSH headings from
    the MeSH index itself (deterministic), keeping only the domain-vocabulary
    additions from the original free-text. Blocks with no resolvable heading keep
    their free-text — that is how non-MeSH jargon survives.
    """
    ix = ix or get_index()
    out = []
    for b in blocks:
        duis = list(b.get("duis") or [])
        if not duis:                     # UI-supplied blocks carry labels only
            for h in b.get("mesh", []):
                d = canonical.resolve_strict(ix, h)
                if d is not None:
                    duis.append(d.dui)
        freetext = list(b.get("freetext", []))
        keep = list(b.get("vocab_freetext", []))
        if strict and duis:
            freetext = canonical.sort_terms(
                ix.strict_terms(sorted(set(duis)), explode=bool(b.get("explode", True))) + keep)
        else:
            freetext = canonical.sort_terms(freetext + keep)
        out.append({
            "name": b.get("name", ""),
            "slot": b.get("slot", "other"),
            "mesh": list(b.get("mesh", [])),
            "freetext": freetext,
            "explode": bool(b.get("explode", True)),
        })
    return out


def _assemble(blocks: list[dict], domains: list[str] | None, *, merge_slots: bool,
              strict: bool, ix) -> dict:
    can = canonical.canonicalize_blocks(blocks, ix, prune=True, merge_slots=merge_slots)
    for b in can["blocks"]:
        b["vocab_freetext"] = _vocab_terms(b, domains, use_model_text=not strict)
    return {
        "blocks": can["blocks"],
        "dropped": can["dropped"],
        "concepts": finalize(can["blocks"], strict=strict, ix=ix),
    }


# ---------------------------------------------------------------- entry points

def build(question: str, *, domains: list[str] | None = None, mode: str = "llm",
          model: str | None = None, api_key: str | None = None,
          extra_context: str = "", seed: int | None = None,
          prompt_version: str | None = None, strict: bool = True,
          fallback: bool = True, merge_slots: bool = False,
          group_by_span: bool = True,
          closure: bool = True, facet_runs: int = 3) -> dict:
    """Synchronous build (evaluation harness / CLI)."""
    mode = _check_mode(mode)
    ix = get_index()
    slate: dict = {}
    raw: dict = {}
    notes = ""
    bad_ids: list[int] = []
    fell_back = False

    if mode == "mesh_only":
        blocks = mesh_only_blocks(question, ix)
    elif mode == "llm":
        raw = map_question(question, domains=domains, model=model, api_key=api_key,
                           extra_context=extra_context, seed=seed,
                           prompt_version=prompt_version)
        notes = raw.get("notes", "")
        blocks = _blocks_from_concepts(raw["concepts"])
    elif mode == "closure":
        fp = facet_pipeline.build(question, ix, model=model, api_key=api_key, k=facet_runs)
        notes = fp["notes"]
        blocks = _blocks_from_closure(fp["concepts"])
    else:
        slate = candidate_slate(question, ix)
        try:
            raw = select_candidates(question, slate, domains=domains, model=model,
                                    api_key=api_key, extra_context=extra_context, seed=seed)
            notes = raw.get("notes", "")
            blocks, bad_ids = _blocks_from_selection(
                raw, slate, group_by_span=group_by_span, closure=closure)
        except OpenRouterError:
            if not fallback:
                raise
            blocks, notes = mesh_only_blocks(question, ix), "LLM selection failed; mesh_only fallback."
            fell_back = True
        if not blocks and fallback:
            blocks = mesh_only_blocks(question, ix)
            notes = (notes + " | empty selection; mesh_only fallback.").strip(" |")
            fell_back = True

    # Slot merging is OFF by default: models label the same content with different
    # slots ("Microglia" came back as population, context AND intervention across
    # models), so merging on that field lets the least reliable part of the answer
    # change what is ANDed vs ORed. Never merge the fallback's unslotted blocks.
    res = _assemble(blocks, domains,
                    merge_slots=(merge_slots and mode == "hybrid" and not fell_back),
                    strict=strict, ix=ix)
    return {**res, "mode": mode, "model": raw.get("model") or model or "",
            "prompt_version": (prompt_version or "v2") if mode == "llm" else mode,
            "notes": notes, "slate": slate, "raw": raw, "invalid_ids": bad_ids}


async def build_async(question: str, *, domains: list[str] | None = None, mode: str = "llm",
                      model: str | None = None, api_key: str | None = None,
                      extra_context: str = "", seed: int | None = None,
                      prompt_version: str | None = None, strict: bool = True,
                      fallback: bool = True, merge_slots: bool = False,
                      group_by_span: bool = True,
                      closure: bool = True, facet_runs: int = 3) -> dict:
    """Async build (FastAPI endpoint)."""
    mode = _check_mode(mode)
    ix = get_index()
    slate: dict = {}
    raw: dict = {}
    notes = ""
    bad_ids: list[int] = []
    fell_back = False

    if mode == "mesh_only":
        blocks = mesh_only_blocks(question, ix)
    elif mode == "llm":
        raw = await map_question_async(question, domains=domains, model=model, api_key=api_key,
                                       extra_context=extra_context, seed=seed,
                                       prompt_version=prompt_version)
        notes = raw.get("notes", "")
        blocks = _blocks_from_concepts(raw["concepts"])
    elif mode == "closure":
        fp = await facet_pipeline.build_async(question, ix, model=model, api_key=api_key,
                                              k=facet_runs)
        notes = fp["notes"]
        blocks = _blocks_from_closure(fp["concepts"])
    else:
        slate = candidate_slate(question, ix)
        try:
            raw = await select_candidates_async(question, slate, domains=domains, model=model,
                                                api_key=api_key, extra_context=extra_context,
                                                seed=seed)
            notes = raw.get("notes", "")
            blocks, bad_ids = _blocks_from_selection(
                raw, slate, group_by_span=group_by_span, closure=closure)
        except OpenRouterError:
            if not fallback:
                raise
            blocks, notes = mesh_only_blocks(question, ix), "LLM selection failed; mesh_only fallback."
            fell_back = True
        if not blocks and fallback:
            blocks = mesh_only_blocks(question, ix)
            notes = (notes + " | empty selection; mesh_only fallback.").strip(" |")
            fell_back = True

    # Slot merging is OFF by default: models label the same content with different
    # slots ("Microglia" came back as population, context AND intervention across
    # models), so merging on that field lets the least reliable part of the answer
    # change what is ANDed vs ORed. Never merge the fallback's unslotted blocks.
    res = _assemble(blocks, domains,
                    merge_slots=(merge_slots and mode == "hybrid" and not fell_back),
                    strict=strict, ix=ix)
    return {**res, "mode": mode, "model": raw.get("model") or model or "",
            "prompt_version": (prompt_version or "v2") if mode == "llm" else mode,
            "notes": notes, "slate": slate, "raw": raw, "invalid_ids": bad_ids}
