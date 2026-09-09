"""
Wires facets.py (self-consistency span voting) + closure.py (deterministic
MeSH closure) into the same {"concepts", "notes", "model"} shape
openrouter_client.map_question(_async) already returns, so backend/app.py's
`resolve_concepts`/frontend contract needs no change to support a new mode.

Mode name: "closure". Kept alongside the existing single-shot "llm" mode
(openrouter_client.py) for side-by-side comparison -- see eval/.
"""
from __future__ import annotations

from . import closure, facets
from .mesh_index import MeshIndex


def concepts_for_compile(concepts: list[dict]) -> list[dict]:
    """
    Convert the UI-facing concept shape (mesh: list of {matched, options,
    selected_dui, ...}) into the plain {name, mesh: [label,...], freetext}
    shape query_builder.compile_search expects -- the same transform
    frontend/app.js:collectConcepts() does before calling /api/compile.
    Shared by the eval harness and any other non-UI caller so this contract
    is defined in exactly one place.
    """
    out = []
    for c in concepts:
        mesh_labels = [m["options"][0]["label"] for m in c.get("mesh", [])
                       if m.get("matched") and m.get("options")]
        oc = {"name": c.get("name", ""), "explode": c.get("explode", True),
              "mesh": mesh_labels, "freetext": c.get("freetext", [])}
        if oc["mesh"] or oc["freetext"]:
            out.append(oc)
    return out


def build(question: str, ix: MeshIndex, *, model: str | None = None,
         api_key: str | None = None, k: int = 3, use_llm: bool = True,
         min_agreement: float = 0.5, explode: bool = True,
         max_freetext: int | None = None) -> dict:
    seg = facets.segment(question, model=model, api_key=api_key, k=k,
                         use_llm=use_llm, min_agreement=min_agreement)
    concepts = closure.build_concepts(ix, question, seg["facets"], explode=explode,
                                      max_freetext=max_freetext)
    notes = (f"facet segmentation mode={seg['mode']} runs={seg['runs']}"
             + (f" agreement>={min_agreement}" if seg["mode"] == "llm_voted" else ""))
    return {"concepts": concepts, "notes": notes,
            "model": model or "closure/heuristic", "segmentation": seg}


async def build_async(question: str, ix: MeshIndex, *, model: str | None = None,
                      api_key: str | None = None, k: int = 3, use_llm: bool = True,
                      min_agreement: float = 0.5, explode: bool = True,
                      max_freetext: int | None = None) -> dict:
    seg = await facets.segment_async(question, model=model, api_key=api_key, k=k,
                                     use_llm=use_llm, min_agreement=min_agreement)
    concepts = closure.build_concepts(ix, question, seg["facets"], explode=explode,
                                      max_freetext=max_freetext)
    notes = (f"facet segmentation mode={seg['mode']} runs={seg['runs']}"
             + (f" agreement>={min_agreement}" if seg["mode"] == "llm_voted" else ""))
    return {"concepts": concepts, "notes": notes,
            "model": model or "closure/heuristic", "segmentation": seg}
