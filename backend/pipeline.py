"""
The question -> concept-blocks pipeline, in three modes.

One entry point shared by the web app and the evaluation harness, so what the
harness measures is exactly what the app does.

  llm        The LLM proposes blocks and headings freely (prompt v2 by default).
             Highest ceiling on judgement, widest output space.
  hybrid     A deterministic MeSH lookup over the question produces a numbered
             candidate slate; the LLM only SELECTS ids from it. A heading that is
             not in the slate cannot enter the query, so the model's whole output
             space is a function of (question, MeSH index) — this is what lifts
             cross-model agreement.
  mesh_only  No LLM at all: every maximal MeSH match in the question becomes a
             block. Trivially model-independent; the baseline and the fallback.

Whatever the mode, the LLM output passes through canonical.canonicalize_blocks
(exact-only resolution, subsumption pruning, canonical ordering) before it can
reach the query builder.
"""
from __future__ import annotations

from . import canonical
from .candidates import candidate_slate, mesh_only_blocks
from .domain_vocab import get_vocab
from .mesh_index import get_index
from .openrouter_client import (
    OpenRouterError,
    map_question,
    map_question_async,
    select_candidates,
    select_candidates_async,
)

MODES = ("llm", "hybrid", "mesh_only")


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


def _blocks_from_selection(selection: dict, slate: dict) -> tuple[list[dict], list[int]]:
    """
    Hybrid selection (candidate ids) -> canonicaliser input.

    Ids outside the slate are dropped: the model is not allowed to smuggle in
    vocabulary, and an out-of-range id is the one failure mode this design has.
    """
    by_id = {c["id"]: c for c in slate.get("candidates", [])}
    blocks, bad_ids = [], []
    used: set[int] = set()
    for b in selection.get("blocks", []):
        labels = []
        for i in b.get("ids", []):
            cand = by_id.get(i)
            if cand is None:
                bad_ids.append(i)
                continue
            if i in used:            # an id may only anchor one block
                continue
            used.add(i)
            labels.append(cand["label"])
        if not labels and not b.get("freetext"):
            continue
        blocks.append({
            "name": b.get("name", ""),
            "slot": b.get("slot", ""),
            "mesh": labels,
            "freetext": b.get("freetext", []),
            "explode": True,
        })
    return blocks, bad_ids


def _vocab_terms(block: dict, domains: list[str] | None) -> list[str]:
    """
    Domain-vocabulary synonyms that apply to a block (deterministic, file-driven).

    Kept separate from the model's free-text so strict mode can drop the model's
    prose while keeping this — it is as reproducible as the MeSH index is.
    """
    hay = " ".join([block.get("name", "")] + list(block.get("mesh", []))
                   + list(block.get("freetext", []))).lower()
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
        b["vocab_freetext"] = _vocab_terms(b, domains)
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
          fallback: bool = True) -> dict:
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
    else:
        slate = candidate_slate(question, ix)
        try:
            raw = select_candidates(question, slate, domains=domains, model=model,
                                    api_key=api_key, extra_context=extra_context, seed=seed)
            notes = raw.get("notes", "")
            blocks, bad_ids = _blocks_from_selection(raw, slate)
        except OpenRouterError:
            if not fallback:
                raise
            blocks, notes = mesh_only_blocks(question, ix), "LLM selection failed; mesh_only fallback."
            fell_back = True
        if not blocks and fallback:
            blocks = mesh_only_blocks(question, ix)
            notes = (notes + " | empty selection; mesh_only fallback.").strip(" |")
            fell_back = True

    # Slot merging is only meaningful for a real selection: the fallback's blocks
    # are all unslotted, and merging them would OR facets meant to be ANDed.
    res = _assemble(blocks, domains, merge_slots=(mode == "hybrid" and not fell_back),
                    strict=strict, ix=ix)
    return {**res, "mode": mode, "model": raw.get("model") or model or "",
            "prompt_version": (prompt_version or "v2") if mode == "llm" else mode,
            "notes": notes, "slate": slate, "raw": raw, "invalid_ids": bad_ids}


async def build_async(question: str, *, domains: list[str] | None = None, mode: str = "llm",
                      model: str | None = None, api_key: str | None = None,
                      extra_context: str = "", seed: int | None = None,
                      prompt_version: str | None = None, strict: bool = True,
                      fallback: bool = True) -> dict:
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
    else:
        slate = candidate_slate(question, ix)
        try:
            raw = await select_candidates_async(question, slate, domains=domains, model=model,
                                                api_key=api_key, extra_context=extra_context,
                                                seed=seed)
            notes = raw.get("notes", "")
            blocks, bad_ids = _blocks_from_selection(raw, slate)
        except OpenRouterError:
            if not fallback:
                raise
            blocks, notes = mesh_only_blocks(question, ix), "LLM selection failed; mesh_only fallback."
            fell_back = True
        if not blocks and fallback:
            blocks = mesh_only_blocks(question, ix)
            notes = (notes + " | empty selection; mesh_only fallback.").strip(" |")
            fell_back = True

    # Slot merging is only meaningful for a real selection: the fallback's blocks
    # are all unslotted, and merging them would OR facets meant to be ANDed.
    res = _assemble(blocks, domains, merge_slots=(mode == "hybrid" and not fell_back),
                    strict=strict, ix=ix)
    return {**res, "mode": mode, "model": raw.get("model") or model or "",
            "prompt_version": (prompt_version or "v2") if mode == "llm" else mode,
            "notes": notes, "slate": slate, "raw": raw, "invalid_ids": bad_ids}
